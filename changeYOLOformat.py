import json
import os

# 클래스명과 클래스 값 매핑 딕셔너리
class_mapping = {'B0': 0, 'B1': 1, 'H0': 2, 'H1': 3, 'M0': 4, 'M1': 5}

# 현재 스크립트 파일의 디렉토리
script_dir = os.path.dirname(__file__)

# 데이터 폴더의 상대 경로
data_folder_relative = 'data'

# 데이터 폴더의 절대 경로
data_folder_absolute = os.path.join(script_dir, data_folder_relative)

def convert_to_yolo_format(subfolder, json_data):
    yolo_lines = []
    
    for annotation in json_data['annotations']:
        # Check if 'bbox' key exists in the annotation
        if 'bbox' not in annotation:
            print(f"Warning: 'bbox' not found in annotation: {annotation}")
            continue

        bbox = annotation['bbox']
        
        # Check if bbox has enough elements
        if len(bbox) != 4:
            print(f"Warning: 'bbox' does not have 4 elements: {bbox}")
            continue

        image_info = json_data['images']
        
        # Extracting bounding box coordinates
        x_center = (bbox[0] + bbox[2]) / 2 / image_info['width']
        y_center = (bbox[1] + bbox[3]) / 2 / image_info['height']
        width = (bbox[2] - bbox[0]) / image_info['width']
        height = (bbox[3] - bbox[1]) / image_info['height']
        
        # 상위 디렉토리 명을 통해 클래스 값을 치환
        # class_name = os.path.basename(os.path.dirname(json_data['images']['file_name']))
        class_value = class_mapping.get(subfolder)

        if class_value == -1:
            print(f"Warning: Unknown class name '{subfolder}' for file {json_data['images']['file_name']}")
            continue

        # Creating YOLO format line
        yolo_line = f"{class_value} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines


def convert_all_json_to_yolo(data_folder):
    # Process each subfolder
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)

        if os.path.isdir(subfolder_path):
            images_folder = os.path.join(subfolder_path, 'images')
            labels_folder = os.path.join(subfolder_path, 'labels')
            os.makedirs(labels_folder, exist_ok=True)

            # Process each JSON file in the images subfolder
            for json_file in os.listdir(images_folder):
                if json_file.endswith('.json'):
                    json_path = os.path.join(images_folder, json_file)

                    with open(json_path, 'r') as file:
                        json_data = json.load(file)

                    # Generate YOLO format lines
                    yolo_lines = convert_to_yolo_format(subfolder,json_data)

                    # Create YOLO format text file in the labels folder
                    txt_file_name = os.path.splitext(json_data['images']['file_name'])[0] + '.txt'
                    txt_file_path = os.path.join(labels_folder, txt_file_name)

                    with open(txt_file_path, 'w') as txt_file:
                        txt_file.write('\n'.join(yolo_lines))

                    print(f"Converted {json_file} to {txt_file_name}")

if __name__ == "__main__":
    data_folder = data_folder_absolute
    convert_all_json_to_yolo(data_folder)
