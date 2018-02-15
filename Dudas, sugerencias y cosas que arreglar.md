# DUDAS
* A la hora de realizar la comprobación de las rutas en la columna *travel_seq* de la tabla **vehicle_trajectories_training_modified**, hay rutas que conectan la intersección C con la barrera de peaje 1 pero no se corresponde con la ruta establecida en la tabla **vehicle_routes_modified**. Es decir, hay rutas en las que pasan de un enlace a otro sin tener en cuenta enlaces intermedios. Un ejemplo para ver esto: 

```sql
SELECT *
FROM vehicle_trajectories_training_modified 
WHERE intersection_id = 'C' AND tollgate_id = 1 AND travel_seq[6].id = '121';
```
* En la tabla donde se visualiza el volumen de tráfico en ventanas de 20 minutos sólo nos proporcionan los días desde el _2016-09-19_ hasta el _2016-10-17_. Hay que tener en cuenta esto a la hora de realizar las predicciones.
* En la tabla **traffic_volume_tollgates_training_modified** se proporciona el atributo **has_etc**, que es importante para un vehículo pasar por la barrera de peaje sin tener que parar a pagar. Sin embargo, este atributo no se si es realmente de utilidad.
* A la hora de predecir es necesario quitar los _outliers_, pero no se si esos _outliers_ son los picos en las gráficas.
* Para hacer las predicciones y tener en cuenta la fecha, se está cogiendo solo la hora. Tengo la duda de si hay que tenerla en cuenta (mirando si es entre semana o fin de semana) o no y, si fuera recomendable tenerla en cuenta, cómo la uso como variable a la hora de introducirla en el modelo.
# SUGERENCIAS
* Se ha dejado al final el volumen de tráfico *out* y el *in* como columnas separadas ya que interesa saber el mismo por separado.

# COSAS QUE ARREGLAR
* Arreglar conjuntos de testeo puesto que estoy cogiendo la tabla que en realidad me proporciona el tiempo promedio de viaje antes de las horas a predecir.

# OBSERVACIONES
* En los días de entrenamiento proporcionados por la competición hay precipitaciones escasas, por lo que los picos que encontramos en los datos de tiempo promedio de viaje no podemos relacionarlos con las precipitaciones.
* A la hora de realizar las predicciones, en la carpeta *testing_phase1* se encuentra la tabla que proporciona el tiempo promedio de viaje en los intervalos de tiempo de dos horas antes de las horas a predecir. De esta forma, se podría unir estos datos a los datos de entrenamiento utilizados en el script, pero dándoles más importancia a las filas de la tabla con los intervalos de tiempo más recientes a las horas a predecir: _You could try building multiple xgboost models, with some of them being limited to more recent data, then weighting those results together. Another idea would be to make a customized evaluation metric that penalizes recent points more heavily which would give them more importance_
