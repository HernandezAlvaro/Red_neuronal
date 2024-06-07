import re
import csv
import pandas as pd 
import torch
import os

def parsear_datos_nodos(datos_nodos):
    nodos = {}
    for linea in datos_nodos:
        partes = linea.split(',')
        if len(partes) >= 4:  # Asegurarse de que haya al menos cuatro partes separadas por comas
            nodo_id = int(partes[1])  # Tomar el segundo elemento como el ID del nodo
            x_str = partes[2].strip()  # Eliminar espacios en blanco alrededor de la cadena
            y_str = partes[3].strip()
            if x_str and y_str:  # Verificar que las cadenas no estén vacías
                try:
                    x = float(x_str)  # Convertir la cadena a un número flotante
                    y = float(y_str)
                    nodos[nodo_id] = {'x': x, 'y': y,}
                except ValueError:
                    print(f"Error al convertir coordenadas en la línea: {linea}")
                    continue
            else:
                print(f"Cadena vacía encontrada en la línea: {linea}")
    return nodos

def encontrar_nodos_adyacentes(nodos):
    for nodo_id, nodo in nodos.items():
        x, y = nodo['x'], nodo['y']
        nodo['arriba'] = -1
        nodo['abajo'] = -1
        nodo['izquierda'] = -1
        nodo['derecha'] = -1
        menor_distancia_arriba = float('inf')
        menor_distancia_abajo = float('inf')
        menor_distancia_izquierda = float('inf')
        menor_distancia_derecha = float('inf')

        for otro_id, otro_nodo in nodos.items():
            if otro_id == nodo_id:
                continue
            otro_x, otro_y = otro_nodo['x'], otro_nodo['y']
            distancia = ((otro_x - x) ** 2 + (otro_y - y) ** 2) ** 0.5

            # Actualizar el nodo más cercano en cada dirección
            if otro_x == x and otro_y > y and distancia < menor_distancia_arriba:
                nodo['derecha'] = otro_id
                menor_distancia_arriba = distancia
            elif otro_x == x and otro_y < y and distancia < menor_distancia_abajo:
                nodo['izquierda'] = otro_id
                menor_distancia_abajo = distancia
            elif otro_y == y and otro_x > x and distancia < menor_distancia_derecha:
                nodo['arriba'] = otro_id
                menor_distancia_derecha = distancia
            elif otro_y == y and otro_x < x and distancia < menor_distancia_izquierda:
                nodo['abajo'] = otro_id
                menor_distancia_izquierda = distancia                           
    return nodos
datos_nodos = [
    'GRID,1,-0.4362678527832,0.4139289855957,0.0',
    'GRID,2,9.5637321472168,0.4139289855957,0.0',
    'GRID,3,19.563732147217,0.4139289855957,0.0',
    'GRID,4,29.563732147217,0.4139289855957,0.0',
    'GRID,5,39.563732147217,0.4139289855957,0.0',
    'GRID,6,49.563732147217,0.4139289855957,0.0',
    'GRID,7,59.563732147217,0.4139289855957,0.0',
    'GRID,8,59.563732147217,-9.5860710144043,0.0',
    'GRID,9,59.563732147217,-19.586071014404,0.0',
    'GRID,10,59.563732147217,-29.586071014404,0.0',
    'GRID,11,59.563732147217,-39.586071014404,0.0',
    'GRID,12,59.563732147217,-49.586071014404,0.0',
    'GRID,13,59.563732147217,-59.586071014404,0.0',
    'GRID,14,59.563732147217,-69.586071014404,0.0',
    'GRID,15,59.563732147217,-79.586071014404,0.0',
    'GRID,16,59.563732147217,-89.586071014404,0.0',
    'GRID,17,59.563732147217,-99.586071014404,0.0',
    'GRID,18,49.563732147217,-99.586071014404,0.0',
    'GRID,19,39.563732147217,-99.586071014404,0.0',
    'GRID,20,29.563732147217,-99.586071014404,0.0',
    'GRID,21,19.563732147217,-99.586071014404,0.0',
    'GRID,22,9.5637321472168,-99.586071014404,0.0',
    'GRID,23,-0.4362678527832,-99.586071014404,0.0',
    'GRID,24,-0.4362678527832,-89.586071014404,0.0',
    'GRID,25,-0.4362678527832,-79.586071014404,0.0',
    'GRID,26,-0.4362678527832,-69.586071014404,0.0',
    'GRID,27,-0.4362678527832,-59.586071014404,0.0',
    'GRID,28,-0.4362678527832,-49.586071014404,0.0',
    'GRID,29,-0.4362678527832,-39.586071014404,0.0',
    'GRID,30,-0.4362678527832,-29.586071014404,0.0',
    'GRID,31,-0.4362678527832,-19.586071014404,0.0',
    'GRID,32,-0.4362678527832,-9.5860710144043,0.0',
    'GRID,33,49.563732147217,-49.586071014404,0.0',
    'GRID,34,39.563732147217,-49.586071014404,0.0',
    'GRID,35,29.563732147217,-49.586071014404,0.0',
    'GRID,36,19.563732147217,-49.586071014404,0.0',
    'GRID,37,9.5637321472168,-49.586071014404,0.0',
    'GRID,38,29.563732147217,-39.586071014404,0.0',
    'GRID,39,29.563732147217,-29.586071014404,0.0',
    'GRID,40,29.563732147217,-19.586071014404,0.0',
    'GRID,41,29.563732147217,-9.5860710144043,0.0',
    'GRID,42,39.563732147217,-29.586071014404,0.0',
    'GRID,43,49.563732147217,-29.586071014404,0.0',
    'GRID,44,39.563732147217,-39.586071014404,0.0',
    'GRID,45,49.563732147217,-39.586071014404,0.0',
    'GRID,46,39.563732147217,-9.5860710144043,0.0',
    'GRID,47,39.563732147217,-19.586071014404,0.0',
    'GRID,48,49.563732147217,-9.5860710144043,0.0',
    'GRID,49,49.563732147217,-19.586071014404,0.0',
    'GRID,50,9.5637321472168,-29.586071014404,0.0',
    'GRID,51,19.563732147217,-29.586071014404,0.0',
    'GRID,52,9.5637321472168,-39.586071014404,0.0',
    'GRID,53,19.563732147217,-39.586071014404,0.0',
    'GRID,54,9.5637321472168,-9.5860710144043,0.0',
    'GRID,55,9.5637321472168,-19.586071014404,0.0',
    'GRID,56,19.563732147217,-9.5860710144043,0.0',
    'GRID,57,19.563732147217,-19.586071014404,0.0',
    'GRID,58,29.563732147217,-89.586071014404,0.0',
    'GRID,59,29.563732147217,-79.586071014404,0.0',
    'GRID,60,29.563732147217,-69.586071014404,0.0',
    'GRID,61,29.563732147217,-59.586071014404,0.0',
    'GRID,62,39.563732147217,-79.586071014404,0.0',
    'GRID,63,49.563732147217,-79.586071014404,0.0',
    'GRID,64,39.563732147217,-89.586071014404,0.0',
    'GRID,65,49.563732147217,-89.586071014404,0.0',
    'GRID,66,39.563732147217,-69.586071014404,0.0',
    'GRID,67,49.563732147217,-69.586071014404,0.0',
    'GRID,68,39.563732147217,-59.586071014404,0.0',
    'GRID,69,49.563732147217,-59.586071014404,0.0',
    'GRID,70,9.5637321472168,-79.586071014404,0.0',
    'GRID,71,19.563732147217,-79.586071014404,0.0',
    'GRID,72,9.5637321472168,-89.586071014404,0.0',
    'GRID,73,19.563732147217,-89.586071014404,0.0',
    'GRID,74,9.5637321472168,-59.586071014404,0.0',
    'GRID,75,9.5637321472168,-69.586071014404,0.0',
    'GRID,76,19.563732147217,-59.586071014404,0.0',
    'GRID,77,19.563732147217,-69.586071014404,0.0',
]
datos_elementos = [
    "CQUAD4,1,1,39,42,44,38,",
    "CQUAD4,2,1,38,44,34,35,",
    "CQUAD4,3,1,11,45,43,10,",
    "CQUAD4,4,1,45,44,42,43,",
    "CQUAD4,5,1,12,33,45,11,",
    "CQUAD4,6,1,33,34,44,45,",
    "CQUAD4,7,1,4,5,46,41,",
    "CQUAD4,8,1,41,46,47,40,",
    "CQUAD4,9,1,40,47,42,39,",
    "CQUAD4,10,1,8,48,6,7,",
    "CQUAD4,11,1,48,46,5,6,",
    "CQUAD4,12,1,10,43,49,9,",
    "CQUAD4,13,1,9,49,48,8,",
    "CQUAD4,14,1,43,42,47,49,",
    "CQUAD4,15,1,49,47,46,48,",
    "CQUAD4,16,1,30,50,52,29,",
    "CQUAD4,17,1,29,52,37,28,",
    "CQUAD4,18,1,38,53,51,39,",
    "CQUAD4,19,1,53,52,50,51,",
    "CQUAD4,20,1,35,36,53,38,",
    "CQUAD4,21,1,36,37,52,53,",
    "CQUAD4,22,1,1,2,54,32,",
    "CQUAD4,23,1,32,54,55,31,",
    "CQUAD4,24,1,31,55,50,30,",
    "CQUAD4,25,1,41,56,3,4,",
    "CQUAD4,26,1,56,54,2,3,",
    "CQUAD4,27,1,39,51,57,40,",
    "CQUAD4,28,1,40,57,56,41,",
    "CQUAD4,29,1,51,50,55,57,",
    "CQUAD4,30,1,57,55,54,56,",
    "CQUAD4,31,1,59,62,64,58,",
    "CQUAD4,32,1,58,64,19,20,",
    "CQUAD4,33,1,62,63,65,64,",
    "CQUAD4,34,1,64,65,18,19,",
    "CQUAD4,35,1,63,15,16,65,",
    "CQUAD4,36,1,65,16,17,18,",
    "CQUAD4,37,1,35,34,66,61,",
    "CQUAD4,38,1,61,66,67,60,",
    "CQUAD4,39,1,60,67,62,59,",
    "CQUAD4,40,1,13,68,33,12,",
    "CQUAD4,41,1,68,66,34,33,",
    "CQUAD4,42,1,15,63,69,14,",
    "CQUAD4,43,1,14,69,68,13,",
    "CQUAD4,44,1,63,62,67,69,",
    "CQUAD4,45,1,69,67,66,68,",
    "CQUAD4,46,1,25,70,72,24,",
    "CQUAD4,47,1,24,72,22,23,",
    "CQUAD4,48,1,70,71,73,72,",
    "CQUAD4,49,1,72,73,21,22,",
    "CQUAD4,50,1,71,59,58,73,",
    "CQUAD4,51,1,73,58,20,21,",
    "CQUAD4,52,1,28,37,74,27,",
    "CQUAD4,53,1,27,74,75,26,",
    "CQUAD4,54,1,26,75,70,25,",
    "CQUAD4,55,1,61,76,36,35,",
    "CQUAD4,56,1,76,74,37,36,",
    "CQUAD4,57,1,60,77,76,61,",
    "CQUAD4,58,1,77,75,74,76,",
    "CQUAD4,59,1,59,71,77,60,",
    "CQUAD4,60,1,71,70,75,77,"
]

# Parsear los datos de los nodos
nodos = parsear_datos_nodos(datos_nodos)

# Encontrar nodos adyacentes
adyacentes = encontrar_nodos_adyacentes(nodos)


def escribir_encabezado_nodos(archivo):
    archivo.write('''Nodos,Vecino(Abajo),Vecino(Arriba),Vecino(Derecha),Vecino(Izquierda),CC,Fuerzas(x),Fuerzas(y),Desplazamientos(x),Desplazamientos(y),Desplazamientos(z),Desplazamientos(rx),Desplazamientos(ry),Desplazamientos(rz)\n''')

def escribir_encabezado_elementos(archivo):
    archivo.write('''Elementos,Nodo_1,Nodo_2,Nodo_3,Nodo_4,Desplazamiento(x1),Desplazamiento(y1),Desplazamiento(x2),Desplazamiento(y2),Desplazamiento(x3),Desplazamiento(y3),Desplazamiento(x4),Desplazamiento(y4),Tensiones(VON),Tensiones(XX1),Tensiones(XX2),Tensiones(YY1),Tensiones(YY2),Tensiones(XY1),Tensiones(XY2),Deformaciones(VON),Deformacones(XX1),Deformaciones(XX2),Deformaciones(YY1),Deformaciones(YY2),Deformaciones(XY1),Deformaciones(XY2)\n''')

def obtener_spc(fem, j):
    for linea in fem:
        partes = linea.strip().split(',') 
        if partes[0] == f'SPC' and partes[1] == '1' and partes[2] == str(j):
            return True
    return False
    
def obtener_fuerza(fem, j):
    for linea in fem:
        partes = linea.strip().split(',')
        if partes[0] == f'FORCE' and partes[1] == '2' and partes[2] == str(j) and partes[3] == '0' and partes[4] == '1.0':
            return ','.join(partes[5:7])
    return '0.0,0.0'

def obtener_resultados(resul, j):
    for linea in resul:
        partes = re.split(r'\s{1,}', linea.strip())
        if partes[0] == str(j):
            return ','.join(partes[1:])
    return '0.0,0.0,0.0,0.0,0.0,0.0'

def nodos_elementos(data,j):
    # Diccionario para guardar las esquinas por elemento
    element_corners = {}

    # Procesar los datos
    for linea in data:
        partes = linea.split(',')
        if partes[1] == str(j):
            return ','.join(partes[3:])
    return 'error'

def buscar_valores(archivo, numeros):
    valores_encontrados_str = ""
    numeros_lista = [num.strip() for num in numeros.split(',') if num.strip()]
    numeros_enteros = [int(num) for num in numeros_lista]
    valores_encontrados = []

    for num in numeros_enteros:
        with open(archivo, 'r') as f:
            reader = csv.reader(f, delimiter=',')  
            next(reader)
            encontrado = False  # Indicador para verificar si se ha encontrado un valor correspondiente
            for row in reader:
                if int(row[0]) >= num:  # Se cambió el operador a '>=' para permitir valores iguales o mayores
                    valor_col_x = row[8]
                    valor_col_y = row[9]
                    valores_encontrados.append((valor_col_x, valor_col_y))
                    encontrado = True
                    break
            if not encontrado:
                valores_encontrados.append(('No encontrado', 'No encontrado'))

    valores_ordenados = [f"{val[0]},{val[1]}" for val in valores_encontrados]
    valores_encontrados_str = ','.join(valores_ordenados)
    return valores_encontrados_str                                                                          

def procesar_datos_nodos(numero_caso):
    with open(f'Data_csv\\Data_{numero_caso}_nodos.csv', 'w') as archivo_nodos, \
         open(f'Data\\Data_{numero_caso}.fem', 'r') as fem, \
         open(f'Data\\Data_{numero_caso}.disp', 'r') as disp:
        
        escribir_encabezado_nodos(archivo_nodos)
        
        for j in range(1, 78):
            fem.seek(0)
            datos_spc = obtener_spc(fem, j)
            fem.seek(0)
            archivo_nodos.write(f'{j},{adyacentes[j]['abajo']},{adyacentes[j]['arriba']},{adyacentes[j]['derecha']},{adyacentes[j]['izquierda']},{1 if datos_spc else 0},')
            fem.seek(0)
            datos_fuerza = obtener_fuerza(fem, j)
            archivo_nodos.write(f'{datos_fuerza},')

            disp.seek(0)
            datos_desplazamiento = obtener_resultados(disp, j)
            archivo_nodos.write(f'{datos_desplazamiento}\n')

def procesar_datos_elementos(numero_caso):           
    with open(f'Data_csv\\Data_{numero_caso}_elementos.csv', 'w') as archivo_elementos, \
         open(f'Data\\Data_{numero_caso}.strs', 'r') as strs, \
         open(f'Data\\Data_{numero_caso}.strn', 'r') as strn:
        
        escribir_encabezado_elementos(archivo_elementos)
        
        for j in range(1,61):
            strs.seek(0)            
            
            datos_tensiones = obtener_resultados(strs, j)
            datos_nodos = nodos_elementos(datos_elementos,j)
            datos_desplazamientos = buscar_valores(f'Data_csv\\Data_{numero_caso}_nodos.csv',datos_nodos)
            archivo_elementos.write(f'{j+77},{datos_nodos}{datos_desplazamientos},{datos_tensiones},')
            
            strn.seek(0)
            datos_deformaciones = obtener_resultados(strn, j)
            archivo_elementos.write(f'{datos_deformaciones}\n')

procesar_datos_nodos(1)