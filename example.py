#!/usr/bin/env python3

''' En este código, se utiliza el extended Kalman Filter porque y=sqrt(vx**2 + vy**2)'''

import rospy
import numpy as np
import kalman.kalman as kalman
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu

class SpeedFilter:
    def __init__(self):
        # Frecuencia de muestreo
        self.dta = 0.1
        self.dt = np.array([[self.dta]])

        # Valores iniciales
        self.xk = np.array([[0.0001, 0.0001, 0.0001]]).T    
        self.uk = np.array([[0.0, 0.0, 0.0]]).T  
        self.Pk = 0.1*np.eye(3)

        # Matrices caracteristicas (estas son constantes, pero Hk no)
        self.Fk = np.eye(3)
        self.Lk = np.eye(3)
        self.Mk = np.array([[1.0]])
        
        # Valores de covarianza 
        self.Qk = (self.dta**2)*np.eye(3)         # De nuestro modelo
        self.Rk = np.array([[0.1]])             # De la medición

        # Inicializando filtro de kalman
        self.speed_filter = kalman.ExtendedFilter(self.xk, self.uk, self.Pk)

        # Subscriptores a los tópicos de velocidad
        rospy.Subscriber("/gps_speed", Float32, self.gps_callback)
        rospy.Subscriber("/imu_accel_vector", Imu, self.imu_callback)

        # Publicador de la velocidad filtrada
        self.vel_filtrada_pub = rospy.Publisher("speed_filtered", Float32, queue_size=10)

    def imu_callback(self, data):
        # En esta funcion se haran las predicciones
        # Primero hace la predicción con lo que ya tenemosf
        f = self.speed_filter.xk + self.dta*self.speed_filter.uk
        
        self.speed_filter.prediction_step(f, self.Fk, self.Lk, self.Qk)

        ax = np.round(data.linear_acceleration.x,1)
        ay = np.round(data.linear_acceleration.y,1)
        az = np.round(data.linear_acceleration.z,1)
        az = 0
        # luego se actualizan los valores
        imu_accel = np.array([[ax, ay, az]]).T 
        self.speed_filter.uk = imu_accel

        


    def gps_callback(self, data):
        # En esta funcion se haran las correcciones
        
        vx = self.speed_filter.xk[0,0]
        vy = self.speed_filter.xk[1,0]
        vz = self.speed_filter.xk[2,0]

        # if vx<0.0001:
        #     vx = 0.0001
        va = (vx**2 + vy**2 + vz**2)**(-1/2)    # constante solo para facilitar el trabajo

        # Realizamos el modelo del sensor
        h = np.sqrt(vx**2 + vy**2 + vz**2)

        # Obtenemos la matriz con respecto a las variables
        self.Hk = np.array([[vx*va, vy*va, vz*va]])

        # Obtenemos la data del sensor
        gps_vel = np.array([[data.data]])*0.27778    # de km/h a m/s
        yk = gps_vel
        
        # Realizamos la correción
        self.speed_filter.correction_step(yk, h, self.Hk,self.Mk, self.Rk)

        


    def run(self):
        # Bucle principal
        while not rospy.is_shutdown():
        
            # Publica la velocidad filtrada
            speed_filtered = np.linalg.norm(self.speed_filter.xk)
            speed_filtered = np.round(speed_filtered,1)
            rospy.loginfo(speed_filtered)
            self.vel_filtrada_pub.publish(speed_filtered)

            # Espera para mantener la frecuencia de muestreo
            rospy.sleep(self.dta)

if __name__ == '__main__':
    # Inicializa el nodo ROS
    rospy.init_node('speed_filter')

    # Crea un objeto ComplementaryFilter
    sf = SpeedFilter()

    # Ejecuta el filtro complementario
    sf.run()

    # Para la ejecución del nodo ROS
    rospy.spin()
