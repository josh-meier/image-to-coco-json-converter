from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json
import os

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]
            if pixel[0] in (12, 13, 14, 15, 16, 17, 18, 19):
                # Check to see if we have created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn"t handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
    
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": "actor",
            "id": value,
            "name": key
        } #notice how the super category and nname are the same here
        category_list.append(category)

    return category_list

def create_video_annotation(id:int, name: str, width = 1920, height = 1080):
    video_info = {
        'id': id,
        'name': name,
        'width': width,
        'height': height 
    }
    return video_info

def create_image_annotation(file_name, width, height, image_id, frame_id, video_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id,
        'frame_id': frame_id,
        'video_id': video_id
    } # we will need to add video id for our specific use case

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id, video_id, instance_id, im_height, im_width):
    min_x, min_y, max_x, max_y = polygon.bounds
    min_x = min(0, min_x)
    min_y = min(0, min_y)

    max_x = max(im_width, max_x)
    max_y = max(im_height, max_y)

    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "id": annotation_id,
        "video_id": video_id,
        "image_id": image_id,
        "category_id": category_id,
        "instance_id": instance_id,
        "bbox": bbox,
        "segmentation": {
            "counts":segmentation,
            "size":[im_height, im_width]
            },
        "area": area,
        "iscrowd": 0,
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "categories": [],
        "videos": [],
        "images": [],
        "annotations": []
    }

    return coco_format

def generate_vis_annotations(path_to_weather_types):
    #we want this fuinction to go through each, weather, each video, and each image 
    # to generate the correstponding annotations which it will dump in a json file at the end


        # so we start with a general path to the main folder and then a list of weather pattern to denote each subfoldr
    # after that we will join those paths and then navigate tpo the instance_seg subfolder where we will see our pictures
    # then we need to iterate through all of the video folders while adding each video to the videos 
    # for each folder go through all of the images
    # fr each image add to images then go through the pixels to generate annotations for seg mask

    print('Generating annotations...\n\n')

    weathers = os.listdir(path_to_weather_types)
    annotation_id = 1

    for weather in weathers:

        ann = get_coco_json_format()

        category_dict = {
            'pedestrian' : 12,
            'rider' : 13,
            'car' : 14,
            'truck' : 15,
            'bus' : 16,
            'train': 17,
            'motorcycle' : 18,
            'bicycle' : 19
        }

        ann['categories'] = create_category_annotation(category_dict)

        # get all videos from within the weather dir
        weather_path = os.path.join(path_to_weather_types, weather, 'instance_seg')
        videos = os.listdir(weather_path)

        for i in range(len(videos)):
            instances = {}
            inst_id = 1
            video_id = i+1
            #get a list of all the image/frames of a video
            image_path = os.path.join(weather_path, videos[i])
            images = os.listdir(image_path)

            #this funcitons assumes standard image  sizes set in the earlier code of 1920x1080
            ann['videos'].append(create_video_annotation(video_id, videos[i]))

            for j in range(len(images)):
                image_id = j+1
                frame = os.path.join(image_path, images[j])
                img = Image.open(frame)

                #add image information to the dictionary
                ann['images'].append(create_image_annotation(os.path.join(videos[i],images[j]),img.width, img.height, image_id, j, video_id))

                sub_masks = create_sub_masks(img, img.width, img.height)

                for k,v in sub_masks.items():
                    polygons, segmentations = create_sub_mask_annotation(v)

                    if polygons != []:
                        if len(polygons) > 1:
                            polygon = MultiPolygon(polygons)
                            segmentation = segmentations
                        else:
                            polygon = polygons[0]
                            segmentation = [np.array(polygons[0].exterior.coords).ravel().tolist()]

                        cur_instance = "-".join(k.split()[1:])

                        if cur_instance in instances:
                            instance_id = instances[cur_instance]
                        else:
                            instances[cur_instance] = inst_id
                            instance_id = inst_id
                            inst_id += 1

                        # polygon, segmentation, image_id, category_id, annotation_id, video_id, instance_id, im_height, im_width
                        annotation = create_annotation_format(polygon, segmentation, image_id, int(k[1:3]), annotation_id, video_id, instance_id, img.height, img.width)
                        ann['annotations'].append(annotation)
                        annotation_id += 1

        
        #dump the annotations for each weather pattern in a json dictionary
        json_path = os.path.join(path_to_weather_types, weather, 'annotations.json')

        with open(json_path, "w") as file:
            json.dump(ann, file, indent=4)

    print('\n\nFinished generating annotations!')


                

if __name__ == '__main__':
    path = '/Data/masaddee/data_trial/0127-101558'

    path_test = '/home/masaddee/Desktop/comps_masaddee/cv4ad_carla/data_collection/example_img'
    generate_vis_annotations(path_test)
    
    
    # img = Image.open(path)

    # img = img.convert("RGB")
    # img.save('/Data/masaddee/data_trial/0126-180341/clear_day/instance_seg/567.png')
    # pixel = img.getpixel((350, 765))
    # pixel_str = str(pixel)

    # print(pixel_str)
    # print(pixel_str[1:3])
    # print("-".join(pixel_str.split()[1:]))
    # print(pixel[0])

    # # Convert to numpy array
    # image_array = np.array(img)

    # # Extract R channel (assuming RGB format)
    # r_channel = image_array[:, :, 0]  # Red is the first channel in RGB

    # # # Print the matrix
    # # for row in r_channel:
    # #     print(" ".join(f"{val:3}" for val in row))

    #     # Write to an external file
    # # output_file = "semantic_tags.txt"
    # # with open(output_file, "w") as f:
    # #     for row in r_channel:
    # #         f.write(" ".join(f"{val:3}" for val in row) + "\n")

    # # print(f"Semantic tag matrix saved to {output_file}")

    # # print(img.height, img.width)

    # # print(os.listdir('/Data/masaddee/data_trial/0126-180341'))

    # # print('\n\n\n')

    # # print(os.listdir('/Data/masaddee/data_trial/0126-180341/clear_day/instance_seg/'))




    # # for x in range(img.width):
    # #     for y in range(img.height):
    # #         print(img.getpixel((x,y)))

    # #     if x == 10 : break

    # # cat_dict = {

    # # }






