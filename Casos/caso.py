import subprocess
import random
import os

def caso(num_casos):
    min = int(1)
    max = int(num_casos + min)
    with open('HW\\Ejercicio1\\Ejercicio1.fem','r') as fem:
        fem.seek(0)
        lineas = fem.readlines()
    for i in range(min,max):
        fuerza ='FORCE,2'
        nombre_archivo = f'Data_{i}.fem'
        ruta_archivo = os.path.join('Data', nombre_archivo)
        with open(ruta_archivo,'w') as fem_nuevo:
            for i, linea in enumerate(lineas):
                if fuerza in linea:        
                    nuevo_valor = str(random.uniform(-10,10))
                    antiguo_valor = '10.0'
                    linea = linea.replace(antiguo_valor,nuevo_valor)   
                    fem_nuevo.write(linea)                                 
                else:
                    fem_nuevo.write(linea)
    return fem_nuevo
