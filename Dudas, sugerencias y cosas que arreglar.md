# DUDAS
* A la hora de realizar la comprobación de las rutas en la columna *travel_seq* de la tabla **vehicle_trajectories_training_modified**, hay rutas que conectan la intersección C con la barrera de peaje 1 pero no se corresponde con la ruta establecida en la tabla **vehicle_routes_modified**. Es decir, hay rutas en las que pasan de un enlace a otro sin tener en cuenta enlaces intermedios. Un ejemplo para ver esto: 

```sql
SELECT *
FROM vehicle_trajectories_training_modified 
WHERE intersection_id = 'C' AND tollgate_id = 1 AND travel_seq[6].id = '121';
```
irando si es entre semana o fin de semana) o no y, si fuera recomendable tenerla en cuenta, cómo la uso como variable a la hora de introducirla en el modelo.



* A la hora de realizar las predicciones de la primera fase del tiempo medio de viaje, solo disponemos del tiempo promedio de viaje de los intervalos de 20 minutos que se encuentran 2 horas antes de los intervalos a predecir, por lo que para las demás vistas ( otros intervalos de 20 minutos) no podemos calcular el tiempo promedio de viaje de las 2 horas previas. -> Pocas filas de entrenamiento (por ejemplo 67)

* No sé si va a ser posible obtener si los días son laborables o no porque no sabemos de donde se cogieron los datos todavia.

* Qué hacer con los valores nulos en la columna de *two_hours_previous*

# SUGERENCIAS
* Se ha dejado al final el volumen de tráfico *out* y el *in* como columnas separadas ya que interesa saber el mismo por separado.

# COSAS QUE ARREGLAR


# OBSERVACIONES
* En los días de entrenamiento proporcionados por la competición hay precipitaciones escasas, por lo que los picos que encontramos en los datos de tiempo promedio de viaje no podemos relacionarlos con las precipitaciones.
* El tipo de vehículo no lo proporciona la competición, por lo que descartamos este atributo.
