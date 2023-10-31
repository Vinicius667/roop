import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from glob import glob
from typing import Union, List
import numpy.typing as npt


def rotate_image(img : npt.NDArray[np.uint8], angle: int) -> npt.NDArray[np.uint8]:
    """
    Rotate image by angle degrees without cutting corners.
    """
    if angle % 360 == 0:
        return img
    
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))


def make_grid(image : npt.NDArray[np.uint8], x_repeat : int, y_repeat : int)-> npt.NDArray[np.uint8]:
    """
    Create a new image by repeating the input image in a grid.
    """
    aux = np.concatenate([image for i in range(x_repeat)], axis = 1)
    aux = np.concatenate([aux for i in range(y_repeat)], axis = 0)
    return aux


def get_water_mark(image_shape: List[int], watermark_text:str, font_size:int, tilt_angle: float, occurrs : Union[None,tuple] = None, raise_on_incomp : bool = False)-> npt.NDArray[np.uint8]:
    """
    Retun watermark image. It does not apply the watermark to the original image because, in case it's used for a video, this water_mark can be a used multiple times without having to create it again for each frame.

    Parameters
    ----------
        image_shape: list of 2 intergers
            shape of the image to be watermarked. If a third dimension is present, it will be ignored.
        watermark_text: str
            text to be used as watermark
        font_size: int
            size of the font
        occurrs: tuple of 2 intergers
            number of times the watermark should be repeated in the x and y direction. This parameter can be omitted and the watermark will be repeated as many times as necessary to cover the entire image.
        tilt_angle: float
            angle of the watermark
        raise_on_incomp: bool
            if True, raise an error if the watermark cannot be repeated the number of times specified in occurrs. If False, the watermark will be repeated as many times as possible.
    """

    h_image, w_image, *_ = image_shape
    font = ImageFont.truetype("./COOPBL.TTF", font_size)


    # Create a blank image with dimensions twice as large as the original image
    image_text = Image.new('L', (w_image * 2, h_image * 2), 0)

    # Get a drawing context
    draw = ImageDraw.Draw(image_text)

    # Draw text
    draw.text((0, 0), watermark_text, font=font, fill=255)
    
    # Get the dimensions of the text image
    left,top,w,h = draw.textbbox(xy= (0,0), text = watermark_text, font=font, spacing=0)

    # Rotate the text image, convert PIL image to numpy array and crop the image
    image_text_np = rotate_image(np.array(image_text)[top:h, :w], angle = tilt_angle)

    # Get the dimensions of the text image
    h_w_single_text = np.array(image_text_np.shape)
    h_single_text, w_single_text = h_w_single_text
        

    # If occurrs is not None, the watermark will be repeated the number of times specified in occurrs
    if occurrs:
        # Times the watermark should be repeated in the x and y direction
        x_repeat, y_repeat = occurrs

        # Add margins to the text image so that it can be repeated the number of times specified in occurrs
        # Margins added to the sides of the text image
        x_margin_px = int((w_image/x_repeat - w_single_text)/2)
        if x_margin_px > 0:
            x_margin = np.zeros((h_single_text, x_margin_px))
            image_text_np = np.concatenate((x_margin, image_text_np, x_margin), axis = 1)
        
        # Margins added to the top and bottom of the text image
        h_single_text, w_single_text = image_text_np.shape
        y_margin_px = int((h_image/y_repeat - h_single_text)/2)
        if y_margin_px > 0:    
            y_margin = np.zeros((y_margin_px, w_single_text))
            image_text_np = np.concatenate((y_margin, image_text_np, y_margin), axis = 0)

        # Make grid with the text image repeated the number of times specified in occurrs
        water_mark = make_grid(image_text_np, int(x_repeat), int(y_repeat))
        # Convert to uint8
        water_mark = water_mark.astype(np.uint8)


        if min(image_shape[0] -  water_mark.shape[0], image_shape[1] - water_mark.shape[1]) < 0:
            msg = "The number of repetitions specified in occurrs is not compatible with the combination of the other parameters."
            if raise_on_incomp:
                raise ValueError(msg)
            else:
                print(msg)
                print("The watermark will be repeated as many times as possible.")

                # Crop the image so that it has the same dimensions as the original image. May not lead to the desired result
                water_mark = water_mark[:min(image_shape[0], water_mark.shape[0]), :min(image_shape[1], water_mark.shape[1])]


        # Get difference between the dimensions of the original image and the watermark
        y_diff = image_shape[0] - water_mark.shape[0]
        x_diff = image_shape[1] - water_mark.shape[1]


        
        # Add margins to the watermark so that it has the same dimensions as the original image
        if y_diff:
            top_margin = np.zeros((y_diff//2, water_mark.shape[1]), dtype = np.uint8)
            bottom_margin = np.zeros((y_diff - y_diff//2, water_mark.shape[1]), dtype = np.uint8)
            water_mark = np.concatenate((top_margin, water_mark, bottom_margin), axis = 0)

        if x_diff:
            left_margin = np.zeros((water_mark.shape[0], x_diff//2), dtype = np.uint8)
            right_margin = np.zeros((water_mark.shape[0], x_diff - x_diff//2), dtype = np.uint8)
            water_mark = np.concatenate((left_margin, water_mark, right_margin), axis = 1, dtype = np.uint8)

    # If occurrs is None, the watermark will be repeated as many times as necessary to cover the entire image
    else:
        y_repeat, x_repeat = np.ceil(np.array((image_shape[0], image_shape[1]))/h_w_single_text)
        water_mark = make_grid(image_text_np, int(x_repeat), int(y_repeat))[:h_image, :w_image]

    # Repeat the watermark in the 3 channels
    water_mark = np.dstack([water_mark for i in range(3)])

    return water_mark

    
def add_water_mark(image: npt.NDArray[np.uint8], watermark:npt.NDArray[np.uint8], alpha )-> npt.NDArray[np.uint8]:
    """
    Wrapper for cv2.addWeighted

    Parameters
    ----------
        image:  
            image to be watermarked
        watermark: watermark image
        alpha: transparency of the watermark
    """
    result_image = cv2.addWeighted(image, 1, watermark, alpha, 0) 
    return result_image


# Examples of use
if __name__ == "__main__":

    ##################################### EXAMPLE 1 #####################################
    # In this examples we will use only one image and vary the parameters

    # Read in the image
    image_path = "./Images/1.jpg"

    image = cv2.imread(image_path)

    watermark_text = 'watermark'

    for font_size in [60, 30]:
        for occurrs in [(3, 2), (2, 3), (1,1)]:
            for transparency in [0.2]:
                for tilt_angle in [30, 45]:
                    # Get watermark
                    water_mark = get_water_mark(image.shape, watermark_text, font_size, tilt_angle, occurrs)

                    # Add watermark to image
                    result_image = add_water_mark(image, water_mark, transparency)

                    # Get basename
                    basename = os.path.basename(image_path)

                    # Save the result
                    path = os.path.join('./Results', f"1_{font_size}_{occurrs[0]}_{occurrs[1]}_{transparency}_{tilt_angle}.jpg")

                    cv2.imwrite(path, result_image)

    #####################################################################################



    ##################################### EXAMPLE 2 #####################################
    # In this example we will use multiple images and fix the parameters
    images = glob('images/*.jpg')
    font_size = 50
    occurrs = (3, 2)
    transparency = 0.2
    tilt_angle = 30

    # Get watermark. Because all the images have the same shape, we can use the same watermark for all of them
    water_mark = get_water_mark((1080, 1920), watermark_text, font_size, tilt_angle, occurrs)

    for image_path in images:
        # Get basename
        basename = os.path.basename(image_path)

        # Path to save the result
        path = os.path.join('./Results', f"{basename}_{font_size}_{occurrs[0]}_{occurrs[1]}_{transparency}_{tilt_angle}.jpg")

        # Read in the image
        image = cv2.imread(image_path)

        # Add watermark to image
        result_image = add_water_mark(image, water_mark, transparency)

        # Save the result
        cv2.imwrite(path, result_image)
        #####################################################################################