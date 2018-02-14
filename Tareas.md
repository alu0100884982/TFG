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

## Próxima reunión
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

### Tareas pendientes
Las tareas a realizar actualmente son las siguientes:
* **Meter la documentación de la nueva base de datos en el documento de carga de la base de datos**.
* **Corregir script de primeras predicciones**.


