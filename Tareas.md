# Tareas TFG
## Reunión 7 de febrero de 2018
### Tareas realizadas
Las tareas realizadas hasta el momento son las siguientes :
* **Creación de un manual de PostgreSQL**, en el que se ha añadido tanto los tipos primitivos como los tipos compuestos de este lenguaje.
* **Creación y carga de las tablas originales de la competición**. Se ha procedido a realizar la carga de una base de datos denominada *tfgdatosoriginales* con los datos originales que nos proporciona la competición KDDCup2017.

* **Creación y carga de las tablas modificadas**. Se ha procedido a realizar la carga de una base de datos denominada *tfgdatosmodificados* con los datos originales que nos proporciona la competición KDDCup2017 pero realizando modificaciones en los tipos de los atributos de las tablas, así como realización de scripts que realizan comprobaciones de que los datos suministrados son correctos.

* **Documentación de la carga de la base de datos**.

* **Realización de un documento con las principales técnicas de machine learning a utilizar**. Se ha creado un documento en el que se describen las principales técnicas de machine learning que se pretenden utilizar sobre los datos. La elección de estas técnicas se ha realizado a partir de las presentaciones de los equipos que obtuvieron mejores resultados al final de la competición.

* **Documentación en Python**. Se ha buscado información acerca de las librerás que nos proporciona Python en cuanto a técnicas de machine learning.

* **Realización de scripts relacionados con las técnicas de machine learning a utilizar**. Se han programado una serie de scripts en los que se muestran ejemplos de aplicación de las técnicas de machine learning que se pretenden utilizar sobre los datos.

* **Creación de gráficas iniciales sobre los datos proporcionados**. Hemos construido una serie de gráficas con el objetivo de familiarizarnos con ellas. Estas gráficas son gráficas de línea y muestran, por cada una de las rutas, el tiempo medio de viaje por horas de cada uno de los días.

### Tareas pendientes
Las tareas a realizar actualmente son las siguientes:
* **Añadir al manual de PostgreSQL los principales meta-comandos y sentencias utilizadas**.Tenemos que añadir los principales comandos y sentencias de PostgreSQL que hemos usado para realizar la carga de la base de datos.
* **Interpretar las rutas de la topología de la red de carreteras proporcionadas con el esquema**. Tenemos que comparar la topología de la red de carreteras que se nos ha proporcionado en la competición con el esquema de las rutas que realmente nos proporcionan en los datos suministrados. Además hay que comprobar que el número de carriles de cada una de las rutas es el correcto, así como la dirección de conducción de las mismas.

* **Realizar gráficas para visualizar mejor los datos proporcionados**. Construir diferentes gráficas para tener una mejor visión de los datos que se nos han proporcionado (gráficas relacionadas con el tiempo meteorológico, con el tráfico, etc.) así como comprobar que las mismas son gráficas lógicas (por ejemplo, que en los dias lluviosos el volumen de tráfico y el tiempo medio de viaje es mayor que en días no lluviosos).

* **Corregir tabla de condiciones meteorológicas**. Hay erratas en algunas columnas de la tabla que contiene las condiciones meteorológicas, por lo que hay que corregirlas.

* **Realizar primeras predicciones**. Hay que realizar las primeras predicciones sobre los datos. Para ello, es necesario que realicemos un primer análisis sobre los datos más relevantes a la hora de realizar las predicciones.

## Reunión 15 de febrero de 2018
### Tareas realizadas
Las tareas realizadas hasta el momento son las siguientes:
* **Corrección de la tabla _weather_data_modified_**. Se ha procedido a eliminar aquellas filas en las que había un valor _999017_ en la columna _wind_direction_ puesto que tiene que haber un valor entre 0 y 360 grados.
* **Interpretación de la topología de la red de carreteras**. Al ser confuso la comparación entre la imagen de la topología de la red de carreteras que se nos proporciona y los datos suministrados en las tablas, nos centraremos en el esquema que representan los datos contenidos en las tablas a la hora de realizar las predicciones. El sentido de circulación que se infiere a partir de los datos es las mencionadas en las tareas de predicción que se deben realizar en la competición:
  * Intersección A a las barreras de peaje 2 & 3.
  * Intersección B a las barreras de peaje 1 & 3.
  * Intersección C a las barreras de peaje 1 & 3.
* **Adición al manual de PostgreSQL los principales meta-comandos y sentencias utilizadas**. Se han añadido los principales meta-comandas y sentencias utilizadas hasta ahora para manejar una base de datos PostgreSQL.
* **Realización de gráficas**. Se ha procedido a construir una gráfica por cada uno de los días de entrenamiento en las que se refleja la evolución del tiempo promedio de viaje en cada una de las rutas por horas. Además, por cada día se ha realizado una gráfica en las que se refleja los diferentes datos meteorológicos por hora. Tambíen se ha realizado una gráfica por cada uno de los días en las que se refleja el volumen de tráfico de entrada en todas las intersecciones y el de salida (menos la barrera de peaje 2 que solo permite dirección de entrada) por horas.
* **Creación de una nueva base de datos para la primera fase de tests del modelo de predicción**. Se ha procedido a crear una nueva base de datos para insertar las tablas relacionadas con la primera fase de testeo del modelo de predicción que se realice. Para ello, se ha añadido la tabla que contiene, por cada una de las rutas, los intervalos de tiempo marcados en verde en la documentación de la competición (los intervalos de tiempo que están justo antes de los intervalos de tiempo a predecir) y se ha añadido la tabla resultado del tiempo promedio de viaje que debe crearse una vez se aplique el modelo de predicción.
* **Creación de un script que crea
### Tareas pendientes
Las tareas a realizar actualmente son las siguientes:
* **Meter la documentación de la nueva base de datos en el documento de carga de la base de datos**.
* **Corregir script de primeras predicciones**.
* **Volver a cargar la tabla de tiempo meteorológico pero, en vez de eliminar las filas con valores erróneos, sustituir los valores erróneos por valores aproximados**. Esto se realiza con el objetivo de no eliminar días de la tabla.
* **Realizar gráficas que contengan tanto el volumen de tráfico como el tiempo promedio de viaje por días**. Se pretende realizar esto con el objetivo de tener otra visualización más global de los datos.
* **Realizar vistas propuestas en la reunión con el objetivo de construir modelos de predicción**. El objetivo es crear diferentes vistas y documentar las vistas realizadas junto con las predicciones  que han tenido lugar sobre esas vistas.
* __Añadir en los scripts proporcionados por la competición la proporción de vehículos que tienen el atributo *has_etc*__ puesto que este atributo puede ser relevante a la hora de realizar los modelos de predicción.
* **Crear un documento genérico de documentación del proyecto**.
* **Realizar el script que construye la tabla resultado de la primera fase de pruebas para la predicción del volumen de tráfico en las barreras de peaje**.
* **Añadir nuevas sentencias utilizadas al manual de PostgreSQL**
* **Añadir documentación de la estructura de las carpetas del repositorio de Github junto con la información de lo que es cada cosa**

## Reunión 22 de febrero de 2018
### Tareas realizadas
Las tareas realizadas hasta el momento son las siguientes:
* **Corregida tabla de datos meteorológicos**. Se han cambiado los valores erróneos en lugar de eliminar las filas que contenían esos valores erróneos con el objetivo de no perder información.
* **Añadida carga de la tabla resultado de las predicciones del volumen de tráfico de la primera fase de pruebas**. Se ha añadido al script de la carga de la base de datos de la primera fase de pruebas el código que carga la tabla resultado de las predicciones del volumen de tráfico de la primera fase de pruebas.
* **Realizadas las gráficas de tiempo promedio de viaje y de volumen de tráfico por días**. Se ha procedido a realizar tanto para el tiempo promedio de viaje (en cada una de las rutas) como para el volumen de tráfico (en las barreras de peaje) las gráficas por día. Los valores de cada día es la media de los valores de ese día (tanto del tiempo promedio de viaje como del volumen de tráfico).
* **Metida la documentación de la base de datos de la primera fase de pruebas de la competición**. Se ha introducido en la documentación de la carga de la base de datos los esquemas de las tablas construidas en la base de datos de la primera fase de pruebas de la competición. También se ha modificado la tabla **volume_table 6_training_20min_avg_volume.csv** para que incluya, en otra columna, la proporción de vehículos que utilizan el dispositivo ETC (Electronic Toll Collection) puesto que hemos considerado que este atributo es importante a la hora de realizar las predicciones.

* **Modificación script _aggregate_volume.py_ para añadir el atributo proportion`_hasetc`_vehicles**. Se ha modificado el script para que, a la hora de agregar los vehículos por cada una de las ventanas de tiempo especificadas y su dirección, también se incluya la proporción de coches que utilizan el dispositivo ETC (Electronic Toll Collection) puesto que hemos considerado que es un atributo relevante para la realización de predicciones.

* **Añadida tabla de volúmenes de tráfico en las barreras de peaje en la base de datos _tfgtest1_**. Se ha añadido la tabla que contiene los volúmenes de tráfico en cada una de las barreras de peaje tanto en la dirección de entrada como en la de salida (menos en la barrera de peaje 2 que no hay dirección de salida) en los intervalos de 2 horas previos a los intervalos de 2 horas a predecir.
* **Realizadas las primeras predicciones**. Se ha procedido a crear vistas sobre las tablas de la base de datos _tfgdatosmodificados_ con el objetivo de obtener, para cada una de las rutas de la competición y para cada uno de los intervalos de tiempo de 20 minutos de los datos de entrenamiento, los datos meteorológicos del día de cada uno de los intervalos junto con el tiempo promedio de viaje y la media de los tiempos promedios de viaje de las dos horas previas al intervalo de tiempo de 20 minutos. Es decir, se ha creado una vista para cada pareja ruta-intervalo de tiempo con los datos mencionados anteriormente. Una vez realizado esto, se ha procedido a crear, en la base de datos _tfgtest1_, el atributo que indica la media de los tiempos promedios de viaje dos horas previas antes de los intervalos de tiempo de 20 minutos a predecir en la primera fase en la tabla resultado de los tiempos promedios de viaje. Por último, se ha escogido una determinada ruta con un intervalo de tiempo de 20 minutos a predecir para coger su vista creada como datos de entrenamiento del modelo de predicción y las filas de la tabla resultado del tiempo promedio de viaje de la primera fase correspondientes a esa ruta y a ese intervalo de tiempo como datos de test para realizar la predicción.

### Tareas pendientes
Las tareas a realizar actualmente son las siguientes:
* **Crear un documento genérico de documentación del proyecto**.
* **Añadir documentación de la estructura de las carpetas del repositorio de Github junto con la información de lo que es cada cosa**
* **Modificación de las vistas realizadas**.
   * Hay que modificar las vistas realizadas previamente (para cada una de las rutas y los intervalos de tiempo de 20 minutos de los datos de entrenamiento). Estas modificaciones consisten en añadir, en lugar de la media de los tiempos promedios de viaje 2 horas antes de los intervalos de tiempo de 20 minutos, la media del tiempo promedio de viaje 20 minutos antes, 40 minutos antes, 60 minutos antes, 80 minutos antes, 100 minutos antes y 120 minutos antes, con el objetivo de que el modelo de predicción vea una tendencia en los datos.
   * Además, la vista de los tiempos meteorológicos por día no es necesario puesto que es importante también tener en cuenta la hora en la que se dan las condiciones meteorológcas, por lo que hay que cambiar la forma en la que se combinan las tablas.
   * No hace falta crear las vistas (o tablas en función de lo que más convenga) para cada una de las rutas e intervalos de tiempo de los datos de entrenamiento, sino sólo aquellas rutas e intervalos de tiempo a predecir, ya que sólo nos interesan los datos de entrenamiento correspondientes a esos valores.
* **Crear modelos de predicción sobre la función _raiz cuadrada_**. El objetivo es crear diferentes modelos de predicción sobre la función _raiz cuadrada_ para tener una mejor visión de cómo se comportan los distintos modelos de predicción a utilizar en los datos de la competición y cómo se miden los distintos errores de valores continuos, con el objetivo de poder utilizarlos de la mejor forma posible.

* **Crear modelos de predicción sobre las vistas (o tablas) creadas**. Para crear estos modelos de predicción, se ha propuesto primero utilizar el propio conjunto de entrenamiento como datos de testeo de los modelos de predicción (con pliegues, por ejemplo) y, una vez que se obtengan resultados satisfactorios, realizar el testeo con el conjunto real de testeo.

## Próxima reunión
### Tareas realizadas
Las tareas realizadas hasta el momento son las siguientes:

### Tareas pendientes
Las tareas a realizar actualmente son las siguientes:
