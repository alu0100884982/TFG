# DUDAS
* A la hora de generar la tabla con las predicciones, tengo la duda de como generar esta tabla. Es decir, no se con certeza como utilizar el script 
de *aggregate* para generarla puesto que no tenemos una tabla de prediccion de "coches que han pasado por las rutas y sus tiempos" en las fechas que queremos predecir el tiempo medio
de viaje para aplicar el script y generar una tabla de predicciones.

* A la hora de realizar la comprobación de las rutas en la columna *travel_seq* de la tabla **vehicle_trajectories_training_modified**, hay rutas que conectan la intersección C con la barrera de peaje 1 pero no se corresponde con la ruta establecida en la tabla **vehicle_routes_modified**. Es decir, hay rutas en las que pasan de un enlace a otro sin tener en cuenta enlaces intermedios. Un ejemplo para ver esto: 

```sql
SELECT *
FROM vehicle_trajectories_training_modified 
WHERE intersection_id = 'C' AND tollgate_id = 1 AND travel_seq[6].id = '121';
```
# SUGERENCIAS


# COSAS QUE ARREGLAR
