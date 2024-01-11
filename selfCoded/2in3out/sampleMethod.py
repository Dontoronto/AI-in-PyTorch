def classify(array):
    if(0 <= array[0]<= 0.3 and 0 <= array[1]<= 0.3):
        return [0,1,0]
    elif((0 <= array[0] < 0.6 and 0.3 < array[1] <= 1) or
         (0.3 < array[0] <= 1 and 0 <= array[1] <  0.6)):
        return [1,0,0]
    elif(0.6 <= array[0]<= 1 and 0.6 <= array[1]<= 1):
        return [0,0,1]
    else:
        return [0,0,0]

