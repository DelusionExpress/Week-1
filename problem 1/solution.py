
def binarySearch(array,start,end,n):
    if start==end:
        return start
    mid = start+ (start-end)//2
    if array[mid]<0:
        if (mid+1<n) and (array[mid+1]>=0):
            return mid
        return binarySearch(array,mid+1,end,n)
    else:
        return binarySearch(array,start,mid-1,n)

def countNumberNegative(array,m,n):
    count = 0 
    endposition = n-1 
    for i in range(m):
        j = endposition
        while j>=0 and array[i][j]>=0 :
            j-=1
        count+=(j+1)
        endposition = j 
        # if array[i][0]>=0:
        #     break
        # endposition = binarySearch(array[i],0,endposition,n)
        # count+=(endposition+1)

    return count
if __name__=='__main__':
    with open('data.txt') as data:
        m,n = [int(x) for x in next(data).split()]
        array = [[int(x) for x in line.split()] for line in data]
    print(countNumberNegative(array,m,n))

