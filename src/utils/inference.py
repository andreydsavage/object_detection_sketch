import cv2

def draw_specific_classes(results, target_classes, colors=None, conf_threshold=0.5):
    """
    Отрисовывает только указанные классы
    
    Args:
        results: результаты YOLO
        target_classes: список ID классов для отрисовки
        colors: словарь {class_id: (B, G, R)}
        conf_threshold: порог уверенности
    """
    if colors is None:
        # Цвета по умолчанию
        colors = {0: (0, 255, 0),  # зеленый для людей
                  25: (0, 0, 255)}  # красный для зонтика
    
    img = results[0].orig_img.copy()
    names = results[0].names
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = box.conf[0].item()
            
            if cls in target_classes and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Получаем цвет для класса
                color = colors.get(cls, (255, 255, 255))
                
                # Отрисовка
                label = f"{names[cls]} {conf:.2f}"
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1, cv2.LINE_AA)
    
    return img

def process_video(video_path, model, target_classes, show = False): 
  # Create a video capture object, in this case we are reading the video from a file
  vid_capture = cv2.VideoCapture(video_path)
  
  if (vid_capture.isOpened() == False):
    print("Error opening the video file")
  # Read fps and frame count
  else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(vid_capture.get(3))     # Width of frame
    frame_height = int(vid_capture.get(4))    # Height of frame

    # init video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('data/output/output.mp4', fourcc, fps, (frame_width, frame_height))

    print('Frames per second : ', fps,'FPS')
  
    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vid_capture.get(7)
    print('Frame count : ', frame_count)
  
  while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = vid_capture.read()
    if ret == True:
      frame = model(frame) # Process model
      frame = draw_specific_classes(frame, target_classes) # Draw bbox
      out.write(frame) # Write video
      if show:
        cv2.imshow('Frame',frame) # Show image
        key = cv2.waitKey(1) # video speed 1 fast, 50 slow
        if key == ord('q'):
            break
    else:
        break
  
  # Release the video capture object
  vid_capture.release()
  out.release()
  cv2.destroyAllWindows()