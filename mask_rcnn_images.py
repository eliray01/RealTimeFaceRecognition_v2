import torch
import torchvision
import cv2
import argparse
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', required=True,
#                     help='path to the input data')
# parser.add_argument('-t', '--threshold', default=0.965, type=float,
#                     help='score threshold for discarding detection')
# args = vars(parser.parse_args())

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def procces(image,a,b,name):
    #image = Image.open(image_path).convert('RGB')
# keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
# transform the image
    image = transform(image)
# add a batch dimension
    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, model, 0.965)
    result = draw_segmentation_map(orig_image, masks, boxes, labels,a,b,name)


    return result


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/elon_musk.mp4")

    while cap.isOpened():
        ret,frame = cap.read()
        frame = procces(frame,(582, 187),(746, 417),'elon')

        #boxes, scores, classes, num = odapi.processFrame(frame)

        #frame_with_rec = frame.copy()

        # for i in range(len(boxes)):
        #     # Class 1 represents human
        #     if classes[i] == 1 and scores[i] > threshold:
        #         box = boxes[i]
        #         #big_a = (box[1],box[0])
        #         #big_b = (box[3],box[2])
        #         # big_c = (box[3],box[2])
        #         # big_d = (box[3],box[0])
        #         print(a,b)
        #         print(big_a, big_b)
        #         if a[0] >= big_a[0] and a[0] <= big_b[0] and a[1] >= big_a[1] and a[1] <= big_b[1] and b[0] >= big_a[0] and b[0] <= big_b[0] and b[1] >= big_a[1] and b[1] <= big_b[1]:
        #             cv2.rectangle(frame_with_rec,(box[1],box[0]),(box[3],box[2]),(0,255,0),-1)
        #             #masked = cv2.bitwise_and(frame, frame, mask=mask)
        #         #else:
        #             #cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)


        #frame = apply_segmentation(frame, color = 'red')
        #final_frame2 = apply_segmentation(frame, frame, color = 'green')
        #final_frame3 = cv2.addWeighted(final_frame, 0.5, final_frame2, 0.5, 0)
        #print("Writing frame {} / {}".format(frame_number, length))
        #output_movie.write(frame)
        cv2.imshow('camera', frame)
        #
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    cap.release()
    cv2.destroyAllWindows()