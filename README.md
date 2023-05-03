# Formulations Imaging

We have developed a 2-stage computer vision model compromising of (1) region proposal using YOLOv5 to identify the liquid formulation in the image (also referred to as “crop detection”) and (2) classification of whether the formulation is stable or not. 

<img width="1170" alt="image" src="https://user-images.githubusercontent.com/56798326/233680769-40ed8222-744c-43a0-9dd0-cb4d9c3e1ef0.png">

To try out this code visit the web app at this [link](http://aket95.pythonanywhere.com/) - please find a [sample image](https://github.com/AniketChitre/stability-computer-vision/blob/master/data/images/opencv_02-03-2023_S118_False_post-pHAdj.png) to test with. The output should look similar to below.

<img width="653" alt="image" src="https://user-images.githubusercontent.com/56798326/235906960-f8fe4f68-9bc0-42b5-952b-5532cc0dfcad.png">
