
# bash scripts/sample.sh imagenet_inpaint_center 5 dbim 1.0
# bash scripts/evaluate.sh imagenet_inpaint_center 5 dbim 1.0

# # # bash scripts/sample.sh imagenet_inpaint_center 5 dbmsolver2
# # # bash scripts/evaluate.sh imagenet_inpaint_center 5 dbmsolver2

bash scripts/sample.sh imagenet_inpaint_center 7 dbmsolver
bash scripts/evaluate.sh imagenet_inpaint_center 7 dbmsolver

bash scripts/sample.sh imagenet_inpaint_center 7 dbim 1.0
bash scripts/evaluate.sh imagenet_inpaint_center 7 dbim 1.0

bash scripts/sample.sh imagenet_inpaint_center 14 dbim 1.0
bash scripts/evaluate.sh imagenet_inpaint_center 14 dbim 1.0

bash scripts/sample.sh imagenet_inpaint_center 14 dbmsolver
bash scripts/evaluate.sh imagenet_inpaint_center 14 dbmsolver

bash scripts/sample.sh imagenet_inpaint_center 20 dbmsolver
bash scripts/evaluate.sh imagenet_inpaint_center 20 dbmsolver

bash scripts/sample.sh imagenet_inpaint_center 20 dbim 0.0
bash scripts/evaluate.sh imagenet_inpaint_center 20 dbim 0.0