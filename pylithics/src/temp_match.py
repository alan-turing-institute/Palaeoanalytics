import cv2
import glob
import numpy as np


def template_matching(templates, image):
	location_index = []
	bboxes = []
	for i, template in enumerate(templates):
		(tW,tH) = template.shape[::-1]
		result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)  # template matching
		threshold = 0.7
		location = np.where(result >= threshold)  # areas where results are >= than threshold value
		if len(location[0]) > 0:
			location_index.append(i)
			for j in zip(*location[::-1]):
				bboxes.append([j[0], j[1], tW , tH])
				cv2.rectangle(image, j, ( j[0] + tW , j[1] + tH ) ,(0, 0, 255), 2)  # draw templates
	print("{} matched locations *before* NMS".format(len(bboxes)))

# apply non-maxima suppression (NMS) to the rectangles
# NMS_boxes = non_max_suppression(np.array(bboxes))
# show the output image
	cv2.imshow("Before NMS", image)
	cv2.waitKey(0)
	return(location_index)

def main():
	template_list = []
	templates = glob.glob("*.png")
	image = cv2.imread("image_and_scales_march_1/images/149.png")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for template_path in templates:
		template_list.append(cv2.imread(template_path, 0)) # load templates
	indexes = template_matching(template_list, image)
	print("{} matched templates.".format(len(indexes)))
if __name__ == "__main__":
    main()