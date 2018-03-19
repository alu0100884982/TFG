<pre>

.
├── cargaBasesDeDatos
│   ├── constraints
│   │   ├── check_links_order_vehicle_trajectories_training_modified.sql
│   │   ├── sum_traveltimes_each_route_compare_total_traveltime.sql
│   │   └── vehicle_trajectories_training_modified_foreign_key.sql
│   ├── scriptTablasModificadas.sql
│   ├── scriptTablasOriginales.sql
│   ├── scriptTablasTest1.sql
│   ├── scriptTablasTraining2.sql
├── datasetsOriginales
│   ├── dataSet_phase2
│   │   ├── 20min_avg_travel_time_training2.csv
│   │   ├── trajectories(table 5)_test2.csv
│   │   ├── trajectories(table_5)_training2.csv
│   │   ├── volume(table 6)_test2.csv
│   │   ├── volume(table 6)_training2.csv
│   │   └── weather (table 7)_2.csv
│   ├── scripts
│   │   ├── aggregate_travel_time.py
│   │   ├── aggregate_volume.py
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
├── documentacion
│   ├── documentacionAdicional
│   │   ├── Estado del arte
│   │   │   ├── ALEIXANDRE - Predicción de tráfico en las carreteras de la red de la Generalitat Valenciana.pdf
│   │   │   ├── ForecastingTrafficTimeSeries.pdf
│   │   │   ├── HowLongWillTheTrafficFlowSeries.pdf
│   │   │   ├── Short-Term-Traffic-Prediction-on-the-UK-Motorway-N_2016_Transportation-Resea.pdf
│   │   │   └── What-Drives-Mobility-Trends--Results-from-Case-Studies_2016_Transportation-R.pdf
│   │   ├──  Machine Learning
│   │   │   ├── machine_learning_a_bayesian_perspective.pdf
│   │   │   ├── machine_learning_mastery_with_python.pdf
│   │   │   └── tips, tricks and hacks that you can use to make better predictions.pdf
│   │   ├── PostgreSQL
│   │   │   └── PostgreSQL.pdf
│   │   └── Python
│   │       ├── CheatSheets
│   │       │   ├── data-visualisation-infographics1.jpg
│   │       │   ├── LearnDataScienceInPython.jpg
│   │       │   ├── Pandas.png
│   │       │   ├── PythonForDataScience.pdf
│   │       │   └── Scikit-Learn.pdf
│   │       ├── CodigosEjemplo
│   │       │   ├── CorrelationsBetweenAttributes.py
│   │       │   ├── datasets
│   │       │   │   ├── pima-indians-diabetes.data.csv
│   │       │   │   └── pima-indians-diabetes.names
│   │       │   ├── DataTypeForEachAttribute.py
│   │       │   ├── DataTypeForEachAttribute.py~
│   │       │   ├── DescriptiveStatistics.py
│   │       │   ├── DescriptiveStatistics.py~
│   │       │   ├── DimensionsOfYourData.py
│   │       │   ├── DimensionsOfYourData.py~
│   │       │   ├── LoadCSVFiles_with_NumPy.py
│   │       │   ├── LoadCSVFiles_with_NumPy.py~
│   │       │   ├── LoadCSVFiles_with_Pandas.py
│   │       │   ├── LoadCSVFiles_with_Pandas.py~
│   │       │   ├── LoadCSVFiles_with_the_PythonStandardLibrary.py
│   │       │   ├── LoadCSVFiles_with_the_PythonStandardLibrary.py~
│   │       │   ├── LoadCSV_from_URL_using_NumPy.py
│   │       │   ├── LoadCSV_from_URL_using_NumPy.py~
│   │       │   ├── LoadCSV_using_Pandas_from_URL.py
│   │       │   ├── LoadCSV_using_Pandas_from_URL.py~
│   │       │   ├── PeekAtYourData.py
│   │       │   └── PeekAtYourData.py~
│   │       ├── python_pandas_tutorial.pdf
│   │       └── ScipyLectures-simple.pdf
│   └── documentacionProyecto
│       ├── arbolDirectorios.md
│       ├── arbolDirectorios.md~
│       ├── Información KDDCup 2017
│       │   ├── Informacion KDDCup 2017.pdf
│       │   └── Route_KDD.pdf
│       ├── Machine Learning
│       │   ├── Machine Learning del proyecto.pdf
│       │   ├── Machine Learning Models.pdf
│       │   └── modelo-arima.pdf
│       ├── PostgreSQL
│       │   ├── Carga de la base de datos con PostgreSQL.pdf
│       │   └── Manual PostgreSQL.pdf
│       └── PresentacionesPDFMejoresResultados
│           ├── Travel Time Prediction
│           │   ├── Task1_1stPlace.pdf
│           │   ├── Task1_2ndPlace.pdf
│           │   └── Task1_3rdPlace.pdf
│           └── Volume Prediction
│               ├── Task2_1stPlace.pdf
│               ├── Task2_2ndPlace.pdf
│               └── Task2_3rdPlace.pdf
├── Dudas, sugerencias y cosas que arreglar.md
├── Dudas, sugerencias y cosas que arreglar.md~
├── graficas
│   ├── grafica_con_promedio_tiempomediodeviaje_ todoslosdias
│   │   ├── ejemplo.py~
│   │   ├── generacion_grafica.py
│   │   ├── generacion_grafica.py~
│   │   └── graficas_pdf.pdf
│   ├── grafica_con_promedio_volumentrafico_todoslosdias
│   │   ├── generacion_grafica.py
│   │   ├── generacion_grafica.py~
│   │   └── graficas_pdf.pdf
│   ├── graficas_iniciales.py
│   ├── graficas_primera_aproximacion
│   │   ├── firstMachineLearningAproximation.py~
│   │   ├── generacion_grafica.py
│   │   ├── generacion_grafica.py~
│   │   ├── grafica_numero_instancias_vistas_ruta-intervalo.py~
│   │   └── graficas_pdf.pdf
│   ├── graficas_tiempomedioviaje_intersecciones_por_día
│   │   ├── graficas_pdf.pdf
│   │   ├── graficas_tiempomedioviaje_intersecciones_por_día.py
│   │   └── graficas_tráfico_intersecciones_por_día.py~
│   ├── graficas_tiempo_meteorologico
│   │   ├── graficas_pdf.pdf
│   │   ├── graficas_tiempo_meteorologico.py
│   │   └── graficas_tiempo_meteorologico.py~
│   └── graficas_volumentrafico_intersecciones_por_dia
│       ├── graficas_pdf.pdf
│       ├── graficas_volumentrafico_intersecciones_por_dia.py
│       └── graficas_volumentrafico_intersecciones_por_dia.py~
├── machineLearningProyecto
│   ├── Anotaciones.md~
│   ├── Aproximaciones
│   │   ├── Tiempo promedio de viaje
│   │   │   ├── 1ª aproximacion
│   │   │   │   ├── borrarTablas_interstollgtime_test1.sql
│   │   │   │   ├── borrarTablas_interstollgtime_test1.sql~
│   │   │   │   ├── firstMachineLearningApproach.py~
│   │   │   │   ├── firstMachineLearningAproximation.py
│   │   │   │   ├── firstMachineLearningAproximation.py~
│   │   │   │   ├── firstMachineLearningModelsOverview.py
│   │   │   │   ├── firstMachineLearningModelsOverview.py~
│   │   │   │   ├── primera_aproximacion.sql~
│   │   │   │   ├── primera_aproximacion_tfgdatosmodificados.sql
│   │   │   │   ├── primera_aproximacion_tfgdatosmodificados.sql~
│   │   │   │   ├── primera_aproximacion_tfgtes1_construccion_tablaconintervalosdoshorasantes.sql~
│   │   │   │   ├── primera_aproximacion_tfgtest1_construccion_tablaconintervalosdoshorasantes.sql
│   │   │   │   ├── primera_aproximacion_tfgtest1_construccion_tablaconintervalosdoshorasantes.sql~
│   │   │   │   ├── primera_aproximacion_tfgtest1_construccion_tablaresultado.sql~
│   │   │   │   ├── primera_aproximacion_tfgtest1.sql
│   │   │   │   ├── primera_aproximacion_tfgtest1.sql~
│   │   │   │   └── primera_aproximacion_tfgtraining2.sql~
│   │   │   └── 2ª aproximacion
│   │   │       ├── secondMachineLearningApproach_LinearRegression.py
│   │   │       └── secondMachineLearningApproach_LinearRegression.py~
│   │   └── Volumen de tráfico
│   │       └── Primera aproximacion
│   │           └── nohaynada.txt
│   ├── borrarVistas.sql~
│   ├── codigosEjemploTecnicasMachineLearning
│   │   ├── ARIMA
│   │   │   ├── ARIMA_example.py
│   │   │   ├── ARIMA_example.py~
│   │   │   ├── createARIMAmodel_example.py
│   │   │   ├── shampoo-sales.csv
│   │   │   ├── shampoo-sales.csv~
│   │   │   ├── tuningParametersARIMA_example.py
│   │   │   └── tuningParametersARIMA_example.py~
│   │   ├── KNN
│   │   │   ├── iris.data
│   │   │   ├── iris.data~
│   │   │   ├── KNN_example.py
│   │   │   └── KNN_example.py~
│   │   ├── LightGBM
│   │   │   ├── adult.csv
│   │   │   ├── XGBoostvsLightGBM.py
│   │   │   └── XGBoostvsLightGBM.py~
│   │   ├── Multiple Layer Perception
│   │   │   ├── wine_classification.py
│   │   │   ├── wine_classification.py~
│   │   │   └── wine_data.csv
│   │   └── XGBoost
│   │       ├── pima-indians-diabetes.csv
│   │       ├── xgboost_example.py
│   │       └── xgboost_example.py~
│   ├── EnlacesDeInteres.md~
│   ├── firsMachineLearningApproach.py~
│   ├── firstMachineLearningApproach.py~
│   ├── primera_aproximacion.sql~
│   ├── Scripts de utilidad
│   │   ├── borrarVistas.sql
│   │   └── borrarVistas.sql~
│   └── vista_1.sql~
├── README.md
├── Tareas.md
└── Tareas.md~

</pre
├── cargaBasesDeDatos
│   ├── constraints
│   │   ├── check_links_order_vehicle_trajectories_training_modified.sql
│   │   ├── sum_traveltimes_each_route_compare_total_traveltime.sql
│   │   └── vehicle_trajectories_training_modified_foreign_key.sql
│   ├── scriptTablasModificadas.sql
│   ├── scriptTablasOriginales.sql
│   ├── scriptTablasTest1.sql
│   ├── scriptTablasTraining2.sql
├── datasetsOriginales
│   ├── dataSet_phase2
│   │   ├── 20min_avg_travel_time_training2.csv
│   │   ├── trajectories(table 5)_test2.csv
│   │   ├── trajectories(table_5)_training2.csv
│   │   ├── volume(table 6)_test2.csv
│   │   ├── volume(table 6)_training2.csv
│   │   └── weather (table 7)_2.csv
│   ├── scripts
│   │   ├── aggregate_travel_time.py
│   │   ├── aggregate_volume.py
│   ├── submission_sample_travelTime.csv
│   ├── submission_sample_volume.csv

* **cargaBasesDeDatos** -> Carpeta que contiene todo lo relacionado con la carga de las bases de datos creadas.
  * **constraints** -> Carpeta que contiene los scripts de comprobación de una serie de restricciones más elaboradas sobre las tablas de las bases de datos.
    * __*check_links_order_vehicle_trajectories_training_modified.sql*__ -> Script de PostgreSQL que realiza, en la tabla *vehicle_trajectories_training_modified*  de la base de datos *tfgdatosmodificados* y para cada uno de los vehículos, la comprobación de que el orden en que recorren los enlaces es el correcto; es decir, comprueba si el orden de los enlaces que recorre corresponde al orden en el que están dispuestos esos enlaces en la tabla *vehicle_routes_modified*.
    * __*sum_traveltimes_each_route_compare_total_traveltime.sql*__ -> Script de PostgreSQL que realiza, en la tabla *vehicle_trajectories_training_modified*  de la base de datos *tfgdatosmodificados* y para cada uno de los vehículos, la diferencia entre la suma de los tiempos que tarda en recorrer cada enlace de la ruta y el tiempo total que aparece en la columna de tiempo total de viaje, con el objetivo de comprobar si estas columnas son iguales.
    * __*vehicle_trajectories_training_modified_foreign_key.sql*__ -> Script de PostgreSQL que realiza, en la tabla *vehicle_trajectories_training_modified*  de la base de datos *tfgdatosmodificados* y para cada uno de los vehículos, la comprobación de que los enlaces que recorren son enlaces correctos.
  * __*scriptTablasModificadas.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgdatosmodificados*.
  * __*scriptTablasOriginales.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgdatosoriginales*.
  * __*scriptTablasTest1.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgtest1*.
  * __*scriptTablasTraining2.sql*__ -> Script de PostgreSQL que realiza la carga de la base de datos *tfgtraining2*.
* **datasetsOriginales** -> Carpeta que contiene los datos originales proporcionados por la competición KDDCup2017.
  * **dataSet_phase2** -> Carpeta que contiene los datos originales de la segunda fase de la competición.
    * __*20min_avg_travel_time_training2.csv*__ -> Archivo que contiene los tiempos promedios de viaje entre los días 18 y 24 de octubre de 2016 agrupados por ruta e intervalo. Estos datos son necesarios para comparar las predicciones realizadas sobre los intervalos a predecir.
    * __*trajectories(table 5)_test2.csv*__ -> Archivo csv que contiene los datos de testeo del tiempo promedio de viaje de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_travel_time.py*, de tal forma que se obtiene el tiempo promedio de viaje en intervalos de 20 minutos dentro de los intervalos de 2 horas antes de los intervalos a predecir en la segunda fase de la competición.
    * __*trajectories(table_5)_training2.csv*__-> Archivo csv que contiene los datos de entrenamiento del tiempo promedio de viaje de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_travel_time.py*, de tal forma que se obtiene el tiempo promedio de viaje en intervalos de 20 minutos.
    * __*volume(table 6)_test2.csv*__ -> Archivo csv que contiene los datos de testeo del volumen de tráfico de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_volume.py*, de tal forma que se obtiene el volumen de tráfico en intervalos de 20 minutos dentro de los intervalos de 2 horas antes de los intervalos a predecir en la segunda fase de la competición.
    * __*volume(table 6)_training2.csv*__ -> Archivo csv que contiene los datos de entrenamiento del volumen de tráfico de la segunda fase de la competición. Es necesario agrupar estos tiempos por intervalos con el script *aggregate_volume.py*, de tal forma que se obtiene el volumen de tráfico en intervalos de 20 minutos.
    * __*weather (table 7)_2.csv*__ -> Archivo csv que contiene los datos meteorológicos de la segunda fase de la competición.
  * **scripts** -> Carpeta que contiene los scripts necesarios para agrupar tanto el tiempo promedio de viaje como el volumen de tráfico en intervalos de 20 minutos.
    * __*aggregate_travel_time.py*__ -> Script de Python proporcionado por la competición para agrupar los tiempos promedios de viaje en intervalos de 20 minutos a partir de la tabla que contiene, por rutas y por vehículo, el tiempo de viaje realizado en un intervalo de tiempo determinado.
    * __*aggregate_volume.py*__ -> Script de Python proporcionado por la competición para agrupar los volúmenes de tráfico en intervalos de 20 minutos a partir de la tabla que contiene, por fechas y tiempos, los vehículos que han pasado por las barreras de peaje y la dirección en la que lo han hecho.
    
     * **Carga de la base de datos con PostgreSQL.pdf** -> Documento que contiene la documentación de la carga de las bases de datos.
