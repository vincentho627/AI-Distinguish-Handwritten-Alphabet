def DetectingLetter(img_array):
    height, width = img_array.shape
    # top = the highest part of the letter
    # bottom = lowest part of the letter
    # left = left part of the letter
    # right = right part of the letter
    # start from max so the for loop finds the minimum the top and left variables
    top, left = height, width
    # starts from min so the for loop finds the maximum the bottom and right
    bottom, right = 0, 0
    # which row on the y axis
    for j in range(height):
        # which col on the x axis
        for i in range(width):
            # when the array value is black, indicating it is a letter
            if img_array[j][i] == 0:
                if j < top:
                    top = j
                if i < left:
                    left = i
                if j > bottom:
                    bottom = j
                if i > right:
                    right = i

    # returns the boxed array
    return img_array[top:bottom+1, left:right+1]


