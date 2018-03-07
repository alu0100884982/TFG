<pre>

.
├── cargaBasesDeDatos
│   ├── Carga de la base de datos con PostgreSQL.pdf
│   ├── constraints
│   │   ├── check_links_order_vehicle_trajectories_training_modified.sql
│   │   ├── sum_traveltimes_each_route_compare_total_traveltime.sql
│   │   └── vehicle_trajectories_training_modified_foreign_key.sql
│   ├── scriptTablasModificadas.sql
│   ├── scriptTablasOriginales.sql
│   ├── scriptTablasTest1.sql
├── datasetsOriginales
│   ├── dataSet_phase2
│   │   ├── trajectories(table 5)_test2.csv
│   │   ├── trajectories(table_5)_training2.csv
│   │   ├── volume(table 6)_test2.csv
│   │   ├── volume(table 6)_training2.csv
│   │   └── weather (table 7)_2.csv
│   ├── scripts
│   │   ├── aggregate_travel_time.py
│   │   └── aggregate_volume.py
│   ├── submission_sample_travelTime.csv
│   ├── submission_sample_volume.csv
│   ├── testing_phase1
│   │   ├── test1_20min_avg_travel_time.csv
│   │   ├── test1_20min_avg_volume.csv
│   │   ├── trajectories(table5)_test1.csv
│   │   ├── volume(table 6)_test1.csv
│   │   └── weather (table 7)_test1.csv
│   └── training
│       ├── links_table3.csv
│       ├── routes_table4.csv
│       ├── trajectories_table5_training_20min_avg_travel_time.csv
│       ├── trajectories_table 5_training.csv
│       ├── volume_table 6_training_20min_avg_volume.csv
│       ├── volume_table 6_training.csv
│       ├── weather (table 7)_training.csv
│       └── weather (table 7)_training.csv~
├── Dudas, sugerencias y cosas que arreglar.md
├── graficas
│   ├── grafica_con_promedio_tiempomediodeviaje_ todoslosdias
│   │   ├── ejemplo.py~
│   │   ├── generacion_grafica.py
│   │   └── graficas_pdf.pdf
│   ├── grafica_con_promedio_volumentrafico_todoslosdias
│   │   ├── generacion_grafica.py
│   │   └── graficas_pdf.pdf
│   ├── graficas_iniciales.py
│   ├── graficas_tiempomedioviaje_intersecciones_por_día
│   │   ├── graficas_pdf.pdf
│   │   ├── graficas_tiempomedioviaje_intersecciones_por_día.py
│   ├── graficas_tiempo_meteorologico
│   │   ├── graficas_pdf.pdf
│   │   ├── graficas_tiempo_meteorologico.py
│   └── graficas_volumentrafico_intersecciones_por_dia
│       ├── graficas_pdf.pdf
│       ├── graficas_volumentrafico_intersecciones_por_dia.py
├── Información KDDCup 2017
│   ├── Informacion KDDCup 2017.pdf
│   └── Route_KDD.pdf
├── InformacionMachineLearning
│   ├── machine_learning_a_bayesian_perspective.pdf
│   ├── Machine Learning Models.pdf
│   └── tips, tricks and hacks that you can use to make better predictions.pdf
├── machineLearningProyecto
│   ├── Anotaciones.md
│   ├── Aproximaciones
│   │   └── Primera aproximacion
│   │       ├── borrarTablas_interstollgtime_test1.sql
│   │       ├── firstMachineLearningApproach.py
│   │       ├── primera_aproximacion_tfgdatosmodificados.sql
│   │       ├── primera_aproximacion_tfgtest1.sql
│   ├── EnlacesDeInteres.md
│   ├── Machine Learning del proyecto.pdf
│   ├── Scripts de utilidad
│   │   ├── borrarVistas.sql
├── PostgreSQL
│   ├── Manual PostgreSQL.pdf
│   └── PostgreSQL.pdf
├── PresentacionesPDFMejoresResultados
│   ├── Travel Time Prediction
│   │   ├── Task1_1stPlace.pdf
│   │   ├── Task1_2ndPlace.pdf
│   │   └── Task1_3rdPlace.pdf
│   └── Volume Prediction
│       ├── Task2_1stPlace.pdf
│       ├── Task2_2ndPlace.pdf
│       └── Task2_3rdPlace.pdf
├── Python
│   ├── Tecnicas de Machine Learning
│   │   ├── LightGBM
│   │   │   ├── adult.csv
│   │   │   ├── XGBoostvsLightGBM.py
│   │   ├── Multiple Layer Perception
│   │   │   ├── wine_classification.py
│   │   │   └── wine_data.csv
│   │   └── XGBoost
│   │       ├── pima-indians-diabetes.csv
│   │       ├── xgboost_example.py
│   └── Tutoriales
│       ├── CheatSheets
│       │   ├── data-visualisation-infographics1.jpg
│       │   ├── LearnDataScienceInPython.jpg
│       │   ├── Pandas.png
│       │   ├── PythonForDataScience.pdf
│       │   └── Scikit-Learn.pdf
│       ├── CodigosEjemplos
│       │   ├── CorrelationsBetweenAttributes.py
│       │   ├── datasets
│       │   │   ├── pima-indians-diabetes.data.csv
│       │   │   └── pima-indians-diabetes.names
│       │   ├── DataTypeForEachAttribute.py
│       │   ├── DescriptiveStatistics.py
│       │   ├── DimensionsOfYourData.py
│       │   ├── LoadCSVFiles_with_NumPy.py
│       │   ├── LoadCSVFiles_with_Pandas.py
│       │   ├── LoadCSVFiles_with_the_PythonStandardLibrary.py
│       │   ├── LoadCSV_from_URL_using_NumPy.py
│       │   ├── LoadCSV_using_Pandas_from_URL.py
│       │   ├── PeekAtYourData.py
│       ├── machine_learning_mastery_with_python.pdf
│       ├── python_pandas_tutorial.pdf
│       └── ScipyLectures-simple.pdf
├── README.md
├── Tareas.md
</pre>


* **cargaBasesDeDatos** -> Carpeta que contiene todo lo relacionado con la carga de las bases de datos creadas.
  * **Carga de la base de datos con PostgreSQL.pdf** -> Documento que contiene la documentación de la carga de las bases de datos.
  * **constraints** -> Carpeta que contiene los scripts de comprobación de una serie de restricciones más elaboradas sobre las tablas de las bases de datos.
    * __*check_links_order_vehicle_trajectories_training_modified.sql*__ -> Script de PostgreSQL que realiza, en la tabla *vehicle_trajectories_training_modified*  de la base de datos *tfgdatosmodificados* y para cada uno de los vehículos, la comprobación de que el orden en que recorren los enlaces es el correcto; es decir, comprueba si el orden de los enlaces que recorre corresponde al orden en el que están dispuestos esos enlaces en la tabla *vehicle_routes_modified*.
    * __*sum_traveltimes_each_route_compare_total_traveltime.sql*__ -> Script de PostgreSQL que realiza, en la tabla *vehicle_trajectories_training_modified*  de la base de datos *tfgdatosmodificados* y para cada uno de los vehículos, la diferencia entre la suma de los tiempos que tarda en recorrer cada enlace de la ruta y el tiempo total que aparece en la columna de tiempo total de viaje, con el objetivo de comprobar si estas columnas son iguales.
    * __*vehicle_trajectories_training_modified_foreign_key.sql*__ -> Script de PostgreSQL que realiza, en la tabla *vehicle_trajectories_training_modified*  de la base de datos *tfgdatosmodificados* y para cada uno de los vehículos, la comprobación de que los enlaces que recorren son enlaces correctos.
  * __*scriptTablasModificadas.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgdatosmodificados*.
  * __*scriptTablasOriginales.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgdatosoriginales*.
  * __*scriptTablasOriginales.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgtest1*.
* **datasetsOriginales** -> Carpeta que contiene los datos originales proporcionados por la competición KDDCup2017.
  * **dataSet_phase2** -> Carpeta que contiene los datos originales de la segunda fase de la competición
    * __*trajectories(table 5)_test2.csv*__ -> Archivo csv que contiene los datos de testeo del tiempo promedio de viaje de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_travel_time.py*, de tal forma que se obtiene el tiempo promedio de viaje en intervalos de 20 minutos dentro de los intervalos de 2 horas antes de los intervalos a predecir en la segunda fase de la competición.
    * __*trajectories(table_5)_training2.csv*__-> Archivo csv que contiene los datos de entrenamiento del tiempo promedio de viaje de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_travel_time.py*, de tal forma que se obtiene el tiempo promedio de viaje en intervalos de 20 minutos.
    * __*volume(table 6)_test2.csv*__ -> Archivo csv que contiene los datos de testeo del volumen de tráfico de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_volume.py*, de tal forma que se obtiene el volumen de tráfico en intervalos de 20 minutos dentro de los intervalos de 2 horas antes de los intervalos a predecir en la segunda fase de la competición.
    * __*volume(table 6)_training2.csv*__ -> Archivo csv que contiene los datos de entrenamiento del volumen de tráfico de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_volume.py*, de tal forma que se obtiene el volumen de tráfico en intervalos de 20 minutos.
    * __*weather (table 7)_2.csv*__ -> Archivo csv que contiene los datos meteorológicos de la segunda fase de la competición.
  * **scripts** -> Carpeta que contiene los scripts necesarios para agrupar tanto el tiempo promedio de viaje como el volumen de tráfico en intervalos de 20 minutos.
    * __*aggregate_travel_time.py*__ -> Script de Python proporcionado por la competición para agrupar los tiempos promedios de viaje en intervalos de 20 minutos a partir de la tabla que contiene, por rutas y por vehículo, el tiempo de viaje realizado en un intervalo de tiempo determinado.
    * __*aggregate_volume.py*__ -> Script de Python proporcionado por la competición para agrupar los volúmenes de tráfico en intervalos de 20 minutos a partir de la tabla que contiene, por fechas y tiempos, los vehículos que han pasado por las barreras de peaje y la dirección en la que lo han hecho.
