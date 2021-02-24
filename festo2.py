import hashlib  
import numpy as np
from collections import defaultdict

def find(start,end,key):
    a1=[x for x in range(48,58)]
    a2=[x for x in range(65,91)]
    a3=[x for x in range(97,123)]

    a = a1+a2+a3

    a = [chr(x) for x in a]
    for x in a:
        for y in a:
            m = start + str(x) + str(y) + end

            hashed = hashlib.md5(m.encode("utf-8")).hexdigest()

            d = hashed[0:len(key)]
            if d == key:
                return hashed, m

    return ""           

def xor7():
    x = []
    z = 0
    while len(x) < 1001:
        z = z+1
        digits = list(str(z))
        flag5 = 0
        flag7 = 0
        for digit in digits:
            if digit == '5':
                flag5 = 1
            if digit == '7':
                flag7 = 1
        if flag5 ^ flag7:
            x.append(z)
    
    return x

def TSP():
    from python_tsp.exact import solve_tsp_dynamic_programming
    matrix = np.genfromtxt(r"C:\\Users\\GabrielPC\\Downloads\\"+'test2.csv', delimiter=',')
    
    a, b = matrix.shape
    for x in range(a):
        for y in range(b):
            if not(matrix[x,y].is_integer()):
                matrix[x,y] = 999
    permutation, distance = solve_tsp_dynamic_programming(matrix)
    
    return permutation, distance

def TSP2():

    from python_tsp.exact import solve_tsp_dynamic_programming
    name = "3_2_delivery_service.csv"
    matrix = np.genfromtxt(r"C:\\Users\\GabrielPC\\Downloads\\"+ name, delimiter=',')
    
    a, b = matrix.shape
    paths=[]

    #Check if path is valid according to restrains
    def isValid(path):
            s = "work r1 r2 r3 r4 r5 c1 c2 c3 c4 c5 home"
            s = s.split()
            c = [s[x] for x in path]
            bag = [0,0,0,0,0]
            for place in c:
                if place[0] == 'r':
                    bag[int(place[1])-1] = 1
                    if sum(bag) > 3:
                        return False
                if place[0] == 'c':
                    if bag[int(place[1])-1] == 0:
                        return False
                    else:
                        bag[int(place[1])-1] = 0
            return True

    class Graph: 
   
        def __init__(self, vertices): 
            # No. of vertices 
            self.V = vertices  
            
            # default dictionary to store graph 
            self.graph = defaultdict(list)  
    
        # function to add an edge to graph 
        def addEdge(self, u, v): 
            self.graph[u].append(v) 

        #Check if the path satisfies the restraining conditions    
    
        '''A recursive function to print all paths from 'u' to 'd'. 
        visited[] keeps track of vertices in current path. 
        path[] stores actual vertices and path_index is current 
        index in path[]'''
        def printAllPathsUtil(self, u, d, visited, path): 
    
            # Mark the current node as visited and store in path 
            visited[u]= True
            path.append(u) 
        
            # If current vertex is same as destination, then print 
            # current path[] 
            if u == d: 
                if len(path) == 12:
                    paths.append(path.copy())
            else: 
                # If current vertex is not destination 
                # Recur for all the vertices adjacent to this vertex 
                for i in self.graph[u]: 
                    if visited[i]== False and isValid(path): 
                        self.printAllPathsUtil(i, d, visited, path) 
                        
            # Remove current vertex from path[] and mark it as unvisited 
            path.pop() 
            visited[u]= False
    
        # Prints all paths from 's' to 'd' 
        def printAllPaths(self, s, d): 
    
            # Mark all the vertices as not visited 
            visited =[False]*(self.V) 
    
            # Create an array to store paths 
            path = [] 
    
            # Call the recursive helper function to print all paths 
            self.printAllPathsUtil(s, d, visited, path) 

        
    
   
    
    # Create a graph given in the above diagram 
    g = Graph(len(matrix)) 
    
    s = 0
    d = 11

    # Create Graph based on matrix
    for x in range(a):
            for y in range(b):
                z = matrix[x,y]
                if z.is_integer():
                    g.addEdge(x,y)

    
    print ("Following are all different paths from % d to % d :" %(s, d)) 
    g.printAllPaths(s, d) 
    
    costs=[]
    for path in paths:
        cost = 0
        for i in range(len(path)-1):
            cost+= matrix[path[i],path[i+1]]
        costs.append(cost.copy())

    y= sorted(range(len(costs)), key=lambda k: costs[k])
    shortestpath=paths[y[0]]
    print("test")
    return 
 
def factors():
    import time
    num = [1]
    c2 = c3 = c5 = 0

    while len(num)<200:
        next = min(7*num[c2], 11*num[c3], 13*num[c5])
        num.append(next)
        if next == 7*num[c2]: c2 += 1
        if next == 11*num[c3]: c3 += 1
        if next == 13*num[c5]: c5 += 1
    print('hello')

def fourier():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.image import imread

    img = cv2.imread("C:\\Users\\GabrielPC\\Desktop\\hidden\\cybersecurity.png",cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    #magnitude_spectrum = np.uint8(fshift)
    fsort = np.sort(f)

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    A= imread("C:\\Users\\GabrielPC\\Desktop\\hidden\\cybersecurity.png")
    f=np.fft.fft2(A)
    f_sort=np.sort(f)
    print("")
    return

def combinations():
    l= "DABC,DCAB,DACB,ADBC,CDAB,ADCB,ABDC,CADB,ACDB,ABCD,CABD,ACBD,DBAC,DCBA,DBCA,BDAC,CDBA,BDCA,BADC,CBDA,BCDA,BACD,CBAD,BCAD"
    y = l.split(sep=",")
    
    eq = {"A": "Ob", "B": "n0", "C":"6u", "D": 0}

    pw = [0,0,0,0]

    pws= []

    for test in y:
        i=0
        for char in test:
            pw[i] = eq[char]
            i+=1
        pws.append(pw.copy())
    
    def startend(order):
        start = ""
        end = ""
        flag = False

        for i in range(len(order)):
            if order[i] == 0:
                flag = True
            elif flag:
                end += order[i]
            else:
                start += order[i]
        return start, end

    key = "a84ba651fd122ef5"
    ps=[]
    for comb in pws:
        start, end = startend(comb)
        p=find(start,end,key)
        if len(p) > 1:
            ps.append(p)
    return 
