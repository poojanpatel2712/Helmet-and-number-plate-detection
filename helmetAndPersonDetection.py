import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def predict(image_path):
    
    img0=image_path
    _ = plt.figure(figsize=(15, 15))
    _ = plt.axis('off')
    _ = plt.imshow(mpimg.imread(img0))

    # for person on bike
    weights0_path = 'person\\yolo-obj_final.weights'
    configuration0_path = 'person\\yolo_pb.cfg'

    probability_minimum = 0.5
    threshold = 0.3

    # In[4]:

    network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
    layers_names0_all = network0.getLayerNames()
    layers_names0_output = [layers_names0_all[i - 1] for i in network0.getUnconnectedOutLayers()]
    labels0 = open('coco.names').read().strip().split('\n')
    print(labels0)

    # In[5]:

    # for helmet
    weights1_path = 'helmet\\yolo-helmet.weights'
    configuration1_path = 'helmet\\yolo-helmet.cfg'

    # In[6]:

    network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)
    layers_names1_all = network1.getLayerNames()
    layers_names1_output = [layers_names1_all[i - 1] for i in network1.getUnconnectedOutLayers()]
    labels1 = open('helmet\\helmet.names').read().strip().split('\n')
    print(labels1)

    # In[7]:

    image_input = cv2.imread(img0)
    blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
    network0.setInput(blob)
    network1.setInput(blob)
    output_from_network0 = network0.forward(layers_names0_output)
    output_from_network1 = network1.forward(layers_names1_output)
    np.random.seed(42)
    colours0 = np.random.randint(0, 255, size=(len(labels0), 3), dtype='uint8')
    colours1 = np.random.randint(0, 255, size=(len(labels1), 3), dtype='uint8')

    print(colours0)
    print(colours1)