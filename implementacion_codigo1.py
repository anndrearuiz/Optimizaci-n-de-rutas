#importamos las librerias que vamos a utilizar
import numpy as np
import pandas as pd

def read_excel_file(filename, sheet_name): #método para leer las coordenadas y la demanda del dataframe creado con los pedidos.
    df = pd.read_excel(filename, sheet_name=sheet_name, header=1) #ponemos header=1 para poner la primera fila de como 
    print(df)
    coordinates = df[['X','Y']].values
    demands = df['Demanda'].values
    return coordinates, demands

def calculate_distance_matrix(coordinates): #Calcular la matriz de distancia entre las coordenadas leidas en el dataframe.
    num_points = len(coordinates)
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = calculate_distance(coordinates, i, j)
    return dist_matrix

def calculate_distance(coordinates, i, j): #Calcular la distancia euclidiana entre los puntos.
    x1, y1 = coordinates[i]
    x2, y2 = coordinates[j]
    return np.sqrt((x1 - x2) **2+ (y1 - y2) **2)

def calculate_total_distance(route, dist_matrix): #Calculate the total distance of a given route using the distance matrix.
    total_distance = 0
    num_points =len(route)
    for i in range(num_points -1):
        current_node = route[i]
        next_node = route[i +1]
        total_distance += dist_matrix[current_node, next_node]
    return total_distance

def nearest_neighbor(dist_matrix, demands, capacity):#Apply the Nearest Neighbor heuristic to find initial routes for VRP.
    num_points = dist_matrix.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    routes = []

    while np.sum(visited) < num_points:
        current_node = 0 # Start at node 0
        current_capacity = 0
        route = [current_node]
        visited[current_node] = True 

        while current_capacity + demands[current_node] <= capacity:
            current = route[-1]
            nearest = None
            min_dist = float('inf')

            for neighbor in np.where(~visited)[0]:
                if demands[neighbor] + current_capacity <= capacity and dist_matrix:
                    nearest = neighbor
                    min_dist = dist_matrix[current, neighbor]
            if nearest is None:
                break


            route.append(nearest)
            visited[nearest] = True
            current_capacity += demands[nearest]
        routes.append(route)
    return routes
