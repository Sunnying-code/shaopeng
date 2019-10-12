
import os

project_path = os.path.dirname(os.path.realpath(__file__))

CKPT_DIR = os.path.join(project_path, "example", "deep")

images_dir = os.path.join(project_path, "example", "mnist_digits_images")



deep_model_path = os.path.join(project_path, "example", "model", "deep")

# test_images_dir = os.path.join(project_path, "example", "images")

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


test_images_dir = os.path.join(project_dir, "src", "example", "images")

# model_path = os.path.join(CKPT_DIR, 'mnist_deep_model.ckpt')

ckpt_path = os.path.join(project_dir, "web", "modules", "mnist", "model", "ckpt")

# new_ckpt_path = os.path.join(project_dir, "src", "model", "ckpt")

files_dir = os.path.join(project_dir, "files")

upload_dir = os.path.join(files_dir, "upload")

for dir_path in [files_dir, upload_dir]:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)