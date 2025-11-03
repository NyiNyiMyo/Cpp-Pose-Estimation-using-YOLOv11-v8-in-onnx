// yolov8_pose_grid.cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;

// Simple IoU for boxes in xyxy format
static float bbox_iou_xyxy(const cv::Rect2f& a, const cv::Rect2f& b) {
    float xx1 = std::max(a.x, b.x);
    float yy1 = std::max(a.y, b.y);
    float xx2 = std::min(a.x + a.width, b.x + b.width);
    float yy2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    return inter / (areaA + areaB - inter + 1e-6f);
}

// NMS
std::vector<int> nms_indices(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float iou_thresh) {
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    while (!idxs.empty()) {
        int cur = idxs[0];
        keep.push_back(cur);
        std::vector<int> rem;
        for (size_t i = 1; i < idxs.size(); ++i) {
            int j = idxs[i];
            if (bbox_iou_xyxy(boxes[cur], boxes[j]) <= iou_thresh) rem.push_back(j);
        }
        idxs.swap(rem);
    }
    return keep;
}

int main() {
    try {
        // -------------------------
        // Settings (edit as needed)
        // -------------------------
        const std::wstring MODEL_PATH = L"yolov8n-pose.onnx"; // set to your YOLOv11 pose onnx path
        const std::string IMAGES_DIR = "pose-tests";         // images folder
        const int INPUT_SIZE = 640;                          // typical YOLOv8 pose export size
        const float CONF_THRES = 0.4f;
        const float NMS_IOU = 0.45f;
        const float KP_CONF_THRES = 0.4f; // draw keypoint if confidence > this
        const int MAX_IMAGES = 8;

        // COCO 17 skeleton pairs
        std::vector<std::pair<int, int>> skeleton = {
            {0,1},{0,2},
            {1,3},{2,4},
            {0,5},{0,6},
            {5,7},{7,9},
            {6,8},{8,10},
            {5,11},{6,12},
            {11,13},{13,15},
            {12,14},{14,16}
        };

        // colors (BGR)
        cv::Scalar COLOR_HEAD(0, 255, 255); // yellow BGR
        cv::Scalar COLOR_ARMS(200, 120, 0); // orange-ish
        cv::Scalar COLOR_BODY(0, 255, 0);
        cv::Scalar COLOR_LEGS(0, 0, 255);

        auto pair_color = [&](int a, int b)->cv::Scalar {
            if (a <= 4 || b <= 4) return COLOR_HEAD;
            if ((a >= 5 && a <= 10) || (b >= 5 && b <= 10)) return COLOR_ARMS;
            if ((a == 5 || a == 6 || a == 11 || a == 12) || (b == 5 || b == 6 || b == 11 || b == 12)) return COLOR_BODY;
            return COLOR_LEGS;
            };

        // -------------------------
        // ONNX Runtime init
        // -------------------------
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "YOLOv8Pose");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        bool use_cuda = false;
        try {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            use_cuda = true;
            std::cout << "[INFO] Attempting to use CUDA Execution Provider\n";
        }
        catch (const std::exception& e) {
            std::cout << "[WARN] CUDA EP not appended (will use CPU): " << e.what() << "\n";
        }

        Ort::Session session(env, MODEL_PATH.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "[INFO] Model loaded (" << (use_cuda ? "GPU" : "CPU") << ")\n";

        // collect images
        std::vector<std::string> all_images;
        for (const auto& entry : fs::directory_iterator(IMAGES_DIR)) {
            if (entry.is_regular_file()) all_images.push_back(entry.path().string());
        }
        if (all_images.empty()) { std::cerr << "No images found in " << IMAGES_DIR << "\n"; return -1; }
        std::shuffle(all_images.begin(), all_images.end(), std::mt19937{ std::random_device{}() });
        if ((int)all_images.size() > MAX_IMAGES) all_images.resize(MAX_IMAGES);

        // input name
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();

        // gather output name(s) - usually single output for yolov8 pose
        size_t out_count = session.GetOutputCount();
        std::vector<const char*> output_names;
        std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
        out_name_ptrs.reserve(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            out_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(out_name_ptrs.back().get());
        }

        std::vector<cv::Mat> vis_images;

        for (const auto& img_path : all_images) {
            cv::Mat img_bgr = cv::imread(img_path);
            if (img_bgr.empty()) { std::cerr << "Failed to read " << img_path << "\n"; continue; }

            int origW = img_bgr.cols, origH = img_bgr.rows;

            // Preprocess: resize to INPUT_SIZE (no padding), BGR->RGB, normalize
            cv::Mat img_rgb;
            cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
            cv::Mat resized;
            cv::resize(img_rgb, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
            resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

            // HWC -> CHW
            std::vector<float> input_tensor_values(1 * 3 * INPUT_SIZE * INPUT_SIZE);
            size_t idx = 0;
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < INPUT_SIZE; ++y) {
                    for (int x = 0; x < INPUT_SIZE; ++x) {
                        cv::Vec3f px = resized.at<cv::Vec3f>(y, x);
                        input_tensor_values[idx++] = px[c];
                    }
                }
            }

            std::array<int64_t, 4> input_dims = { 1, 3, INPUT_SIZE, INPUT_SIZE };
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

            // Run
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, output_names.data(), (int)output_names.size());

            if (output_tensors.empty()) {
                std::cerr << "Model returned no outputs for " << img_path << "\n";
                vis_images.push_back(img_bgr);
                continue;
            }

            // Expecting single output (common): shape either [1, C, N] or [1, N, C]
            Ort::Value& ot = output_tensors[0];
            auto info = ot.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            size_t elem_count = info.GetElementCount();
            float* out_ptr = ot.GetTensorMutableData<float>();

            // Determine channels/per-detect length and N (num anchors/predictions)
            int64_t dim0 = shape.size() > 0 ? shape[0] : 0;
            int64_t dim1 = shape.size() > 1 ? shape[1] : 0;
            int64_t dim2 = shape.size() > 2 ? shape[2] : 0;

            int64_t C = 0, N = 0;
            // Case A: [1, C, N]
            if (dim0 == 1 && dim1 > 0 && dim2 > 0) {
                C = dim1; N = dim2;
            }
            // Case B: [1, N, C]
            else if (dim0 == 1 && dim1 > 0 && dim2 > 0) {
                // same as above; handled
                C = dim2; N = dim1;
            }
            else {
                // fallback: try to infer by dividing
                if (dim0 == 1 && elem_count > 0) {
                    // assume C = 56 typical for pose; try 56
                    const int guessedC = 56;
                    if (elem_count % guessedC == 0) { C = guessedC; N = elem_count / guessedC; }
                    else { C = static_cast<int64_t>(elem_count); N = 1; }
                }
            }

            if (C <= 5) {
                std::cerr << "Unexpected output channel count C=" << C << " for " << img_path << "\n";
                vis_images.push_back(img_bgr);
                continue;
            }

            // Build a per-detection matrix pred[n][C] from out_ptr. Need to handle storage format.
            // If original shape was [1, C, N], out_ptr is contiguous as C*N with C moving fastest? ONNX gives row-major: dims as given.
            // We'll read generically: if shape == [1, C, N] then out_ptr[(c * N) + n]; if [1, N, C] then out_ptr[(n * C) + c].
            bool is_C_N = (dim1 == C && dim2 == N); // shape was [1,C,N]
            std::vector<std::vector<float>> pred; pred.assign((size_t)N, std::vector<float>((size_t)C));
            if (is_C_N) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t n = 0; n < N; ++n) {
                        pred[(size_t)n][(size_t)c] = out_ptr[(size_t)(c * N + n)];
                    }
                }
            }
            else {
                // assume format [1, N, C]
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        pred[(size_t)n][(size_t)c] = out_ptr[(size_t)(n * C + c)];
                    }
                }
            }

            // Parse predictions:
            // typical layout: [x, y, w, h, score, kp0_x, kp0_y, kp0_conf, ..., kp16_x, kp16_y, kp16_conf]
            std::vector<cv::Rect2f> boxes; boxes.reserve(N);
            std::vector<float> scores; scores.reserve(N);
            std::vector<std::vector<float>> kpts; // per-detection raw kpts flatten (K*3)
            int num_kps = (int)((C - 5) / 3); // (C-5) should be 51 for 17 keypoints

            for (int64_t i = 0; i < N; ++i) {
                float obj_score = pred[i][4];
                if (obj_score <= CONF_THRES) continue;

                float cx = pred[i][0];
                float cy = pred[i][1];
                float w = pred[i][2];
                float h = pred[i][3];

                // convert xywh -> xyxy (on resized image coordinates or normalized)
                float x1 = cx - w / 2.0f;
                float y1 = cy - h / 2.0f;
                float x2 = cx + w / 2.0f;
                float y2 = cy + h / 2.0f;

                // store as Rect2f (x,y,width,height)
                boxes.emplace_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
                scores.push_back(obj_score);

                // keypoints
                std::vector<float> kp;
                if (num_kps > 0) {
                    kp.resize(num_kps * 3);
                    for (int k = 0; k < num_kps; ++k) {
                        int base = 5 + k * 3;
                        if (base + 2 < (int)C) {
                            kp[k * 3 + 0] = pred[i][base + 0];
                            kp[k * 3 + 1] = pred[i][base + 1];
                            kp[k * 3 + 2] = pred[i][base + 2];
                        }
                        else {
                            kp[k * 3 + 0] = 0; kp[k * 3 + 1] = 0; kp[k * 3 + 2] = 0;
                        }
                    }
                }
                kpts.push_back(std::move(kp));
            }

            // If none survived threshold
            if (boxes.empty()) {
                vis_images.push_back(img_bgr);
                continue;
            }

            // Detect if coords are normalized (<=1) or already pixel-scale to INPUT_SIZE
            bool normalized_coords = false;
            float max_box_val = 0.0f;
            for (const auto& b : boxes) {
                max_box_val = std::max(max_box_val, std::max(std::max(std::abs(b.x), std::abs(b.y)), std::max(std::abs(b.width + b.x), std::abs(b.height + b.y))));
            }
            if (max_box_val <= 1.01f) normalized_coords = true;

            // If normalized, scale boxes to resized pixels
            if (normalized_coords) {
                for (auto& b : boxes) {
                    b.x *= INPUT_SIZE;
                    b.y *= INPUT_SIZE;
                    b.width *= INPUT_SIZE;
                    b.height *= INPUT_SIZE;
                }
                for (auto& kp : kpts) {
                    for (size_t t = 0; t + 1 < kp.size(); t += 3) {
                        kp[t + 0] *= INPUT_SIZE; // x
                        kp[t + 1] *= INPUT_SIZE; // y
                        // kp[t+2] remains confidence
                    }
                }
            }

            // Apply NMS (works in resized-image pixel coords)
            std::vector<int> keep = nms_indices(boxes, scores, NMS_IOU);

            // Build visualization image
            cv::Mat vis = img_bgr.clone();

            float scale_x = static_cast<float>(origW) / static_cast<float>(INPUT_SIZE);
            float scale_y = static_cast<float>(origH) / static_cast<float>(INPUT_SIZE);

            for (int idx_keep : keep) {
                if (idx_keep < 0 || idx_keep >= (int)boxes.size()) continue;
                cv::Rect2f& b = boxes[idx_keep];

                // scale box to original image
                int x1 = std::max(0, (int)std::round(b.x * scale_x));
                int y1 = std::max(0, (int)std::round(b.y * scale_y));
                int x2 = std::min(origW - 1, (int)std::round((b.x + b.width) * scale_x));
                int y2 = std::min(origH - 1, (int)std::round((b.y + b.height) * scale_y));

                // draw bbox
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 4);

                // label/conf
                std::ostringstream ss; ss << std::fixed << std::setprecision(2) << scores[idx_keep];
                cv::putText(vis, ss.str(), cv::Point(x1, std::max(10, y1 - 10)), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 3);

                // draw keypoints if available
                if (idx_keep < (int)kpts.size() && !kpts[idx_keep].empty()) {
                    const std::vector<float>& kp = kpts[idx_keep];
                    int K = (int)kp.size() / 3;
                    std::vector<cv::Point> scaled_pts(K, cv::Point(-1, -1));
                    std::vector<float> kp_conf(K, 0.0f);

                    for (int k = 0; k < K; ++k) {
                        float kx = kp[k * 3 + 0];
                        float ky = kp[k * 3 + 1];
                        float kc = kp[k * 3 + 2];

                        // Map to original image size
                        int px = (int)std::round(kx * scale_x);
                        int py = (int)std::round(ky * scale_y);
                        scaled_pts[k] = cv::Point(px, py);
                        kp_conf[k] = kc;
                        if (kc > KP_CONF_THRES) {
                            // draw small circle - color choose by head color for visibility; more refined coloring below for skeleton
                            cv::circle(vis, scaled_pts[k], 6, COLOR_HEAD, -1);
                        }
                    }

                    // draw skeleton lines with pair color
                    for (const auto& pr : skeleton) {
                        int a = pr.first, b2 = pr.second;
                        if (a < 0 || a >= K || b2 < 0 || b2 >= K) continue;
                        if (kp_conf[a] > KP_CONF_THRES && kp_conf[b2] > KP_CONF_THRES) {
                            cv::Scalar col = pair_color(a, b2);
                            cv::line(vis, scaled_pts[a], scaled_pts[b2], col, 5, cv::LINE_AA);
                        }
                    }
                }
            }

            vis_images.push_back(vis);
        } // end images loop

        // -------------------------
        // Display grid (2x4)
        // -------------------------
        if (vis_images.empty()) {
            std::cerr << "No visualizations to show\n";
            return -1;
        }

        int rows = 2, cols = 4;
        int cell_w = 256, cell_h = 256; // choose larger cells for pose detail
        cv::Mat grid(rows * cell_h, cols * cell_w, vis_images[0].type(), cv::Scalar(0, 0, 0));

        for (size_t i = 0; i < vis_images.size(); ++i) {
            int r = i / cols;
            int c = i % cols;
            cv::Mat img = vis_images[i];

            float scale = std::min((float)cell_w / img.cols, (float)cell_h / img.rows);
            int new_w = std::max(1, (int)std::round(img.cols * scale));
            int new_h = std::max(1, (int)std::round(img.rows * scale));
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h));

            cv::Mat canvas(cell_h, cell_w, img.type(), cv::Scalar(0, 0, 0));
            int offset_x = (cell_w - new_w) / 2;
            int offset_y = (cell_h - new_h) / 2;
            resized.copyTo(canvas(cv::Rect(offset_x, offset_y, new_w, new_h)));

            cv::Rect roi(c * cell_w, r * cell_h, cell_w, cell_h);
            canvas.copyTo(grid(roi));
        }

        cv::namedWindow("YOLOv8-v11 Pose ONNX Grid", cv::WINDOW_AUTOSIZE);
        cv::imshow("YOLOv8-v11 Pose ONNX Grid", grid);
        cv::waitKey(0);
        cv::destroyAllWindows();

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
