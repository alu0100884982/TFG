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
│   │       │   ├── DescriptiveStatistics.py
│   │       │   ├── DimensionsOfYourData.py
│   │       │   ├── LoadCSVFiles_with_NumPy.py
│   │       │   ├── LoadCSVFiles_with_Pandas.py
│   │       │   ├── LoadCSVFiles_with_the_PythonStandardLibrary.py
│   │       │   ├── LoadCSV_from_URL_using_NumPy.py
│   │       │   ├── LoadCSV_using_Pandas_from_URL.py
│   │       │   ├── PeekAtYourData.py
│   │       ├── python_pandas_tutorial.pdf
│   │       └── ScipyLectures-simple.pdf
│   └── documentacionProyecto
│       ├── arbolDirectorios.md
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
├── graficas
│   ├── grafica_con_promedio_tiempomediodeviaje_ todoslosdias
│   │   ├── generacion_grafica.py
│   │   └── graficas_pdf.pdf
│   ├── grafica_con_promedio_volumentrafico_todoslosdias
│   │   ├── generacion_grafica.py
│   │   └── graficas_pdf.pdf
│   ├── graficas_iniciales.py
│   ├── graficas_primera_aproximacion_tiempopromedioviaje
│   │   ├── generacion_grafica.py
│   │   └── graficas_pdf.pdf
│   ├── graficas_tiempomedioviaje_intersecciones_por_día
│   │   ├── graficas_pdf.pdf
│   │   ├── graficas_tiempomedioviaje_intersecciones_por_día.py
│   ├── graficas_tiempo_meteorologico
│   │   ├── graficas_pdf.pdf
│   │   ├── graficas_tiempo_meteorologico.py
│   └── graficas_volumentrafico_intersecciones_por_dia
│       ├── graficas_pdf.pdf
│       ├── graficas_volumentrafico_intersecciones_por_dia.py
├── machineLearningProyecto
│   ├── Aproximaciones
│   │   ├── Tiempo promedio de viaje
│   │   │   ├── 1ª aproximacion
│   │   │   │   ├── borrarTablas_interstollgtime_test1.sql
│   │   │   │   ├── firstMachineLearningAproximation.py
│   │   │   │   ├── firstMachineLearningAproximation_otherMAPE.py
│   │   │   │   ├── primera_aproximacion_tfgdatosmodificados.sql
│   │   │   │   ├── primera_aproximacion_tfgtest1_construccion_tablaconintervalosdoshorasantes.sql
│   │   │   │   ├── primera_aproximacion_tfgtest1.sql
│   │   │   └── 2ª aproximacion
│   │   │       ├── secondMachineLearningApproach_ARIMA.py
│   │   └── Volumen de tráfico
│   │       └── Primera aproximacion
│   │           └── nohaynada.txt
│   ├── codigosEjemploTecnicasMachineLearning
│   │   ├── ARIMA
│   │   │   ├── ARIMA_example.py
│   │   │   ├── shampoo-sales.csv
│   │   │   ├── tuningParametersARIMA_example.py
│   │   ├── KNN
│   │   │   ├── iris.data
│   │   │   ├── KNN_example.py
│   │   ├── LightGBM
│   │   │   ├── adult.csv
│   │   │   ├── XGBoostvsLightGBM.py
│   │   ├── Multiple Layer Perception
│   │   │   ├── wine_classification.py
│   │   │   └── wine_data.csv
│   │   └── XGBoost
│   │       ├── pima-indians-diabetes.csv
│   │       ├── xgboost_example.py
│   ├── Scripts de utilidad
│   │   ├── borrarVistas.sql
├── README.md
├── Tareas.md

</pre>


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
  * **submission_sample_travelTime.csv** -> Fichero ejemplo que contiene la plantilla utilizada para incluir las predicciones realizadas del tiempo promedio de viaje.
  * **submission_sample_volume.csv** -> Fichero ejemplo que contiene la plantilla utilizada para incluir las predicciones realizadas del volumen de tráfico.
  * **testing_phase1** -> Carpeta que contiene los datos de testeo de la primera fase de la competición.
    * __*test1_20min_avg_travel_time.csv*__ -> Fichero que agrupa el tiempo promedio de viaje por rutas e intervalos de tiempo de las 2 horas antes de los intervalos a predecir.
    * __*test1_20min_avg_volume.csv*__ -> Fichero que agrupa el volumen de tráfico por rutas e intervalos de tiempo de las 2 horas antes de los intervalos a predecir.
    * __*trajectories(table5)_test1.csv*__ -> Fichero que contiene los vehículos que han pasado por una de las rutas contempladas en los datos de la competición en las dos horas previas a los intervalos a predecir junto al tiempo que tarda en recorrer cada uno de los enlaces de la ruta y el tiempo promedio del viaje realizado. Se utiliza para obtener el fichero _test1_20min_avg_travel_time.csv_.
    * __*volume(table 6)_test1.csv*__ -> Fichero que contiene los vehículos que han pasado por una de las barreras de peaje contempladas en la competición en las dos horas previas a los intervalos a predecir junto con la dirección en la que ha atravesado la barrera de peaje y otras características. Se utiliza para obtener el fichero _test1_20min_avg_volume.csv_.
    * __*weather (table 7)_test1.csv*__ -> Fichero que contiene los datos meteorológicos cada 3 horas de los días a predecir.
  * **training** -> Carpeta que contiene los datos de entrenamiento de la primera fase de la competición.
    * __*links_table3.csv*__ -> Fichero que contiene cada uno de los enlaces que forman las rutas de la competición junto con sus características.
    * __*routes_table4.csv*__ -> Fichero que contiene cada una de las rutas de la competición junto con la secuencia de enlaces que la forman.
    * __*trajectories_table5_training_20min_avg_travel_time.csv*__ -> Fichero que agrupa el tiempo promedio de viaje por rutas e intervalos de tiempo de entrenamiento.
    * __*trajectories_table 5_training.csv*__ -> Fichero que contiene los vehículos que han pasado por una de las rutas contempladas en los datos de la competición en los intervalos de tiempo de entrenamiento junto al tiempo que tarda en recorrer cada uno de los enlaces de la ruta y el tiempo promedio del viaje realizado. Se utiliza para obtener el fichero _trajectories_table5_training_20min_avg_travel_time.csv_.
    * __*volume_table 6_training_20min_avg_volume.csv*__ ->  Fichero que agrupa el volumen de tráfico por rutas e intervalos de tiempo de entrenamiento.
    * __*volume_table 6_training.csv*__ -> Fichero que contiene los vehículos que han pasado por una de las barreras de peaje contempladas en la competición en los intervalos de tiempo de entrenamiento junto con la dirección en la que ha atravesado la barrera de peaje y otras características. Se utiliza para obtener el fichero _volume_table 6_training_20min_avg_volume.csv_.
    * __*weather (table 7)_training.csv*__ -> Fichero que contiene los datos meteorológicos cada 3 horas de los días de entrenamiento.
* **documentacion** -> Carpeta que contiene toda la documentación del TFG.
  * **documentacionAdicional** -> Carpeta que contiene los datos de entrenamiento de la primera fase de la competición.
     * __*Estado del arte*__ -> Fichero que contiene los documentos escogidos para representar el estado del arte del tema tratado en el TFG.  
        * __*ALEIXANDRE - Predicción de tráfico en las carreteras de la red de la Generalitat Valenciana.pdf*__ -> Trabajo de Fin de Grado acerca de la predicción de tráfico en las carreteras de la red de la Generalitat Valenciana.
        * __*ForecastingTrafficTimeSeries.pdf*__ -> Artículo acerca de la predicción de tráfico.
        * __*HowLongWillTheTrafficFlowSeries.pdf*__ -> Artículo en el que se estudia la cuestión acerca del tiempo de validez en que se mantiene eficaz una serie histórica de tiempos de flujo del tráfico para predecir el futuro.
        * __*Short-Term-Traffic-Prediction-on-the-UK-Motorway-N_2016_Transportation-Resea.pdf*__ -> Artículo en el que se realizan estimaciones a corto plazo (15 minutos en el futuro) con la información histórica del tráfico de la red de autopistas del Reino Unido utilizando redes neuronales, de tal forma que esto permita reducir la congestión del transporte mediante la mejora de sistemas inteligentes de transporte utilizados para controlar el tráfico para que realicen decisiones proactivas sobre la red de carreteras (anticiparse al inicio de la congestión del tráfico)
        * __*What-Drives-Mobility-Trends--Results-from-Case-Studies_2016_Transportation-R.pdf*__ -> Artículo en el que se estudian las tendencias de movilidad urbana en Paris, Santiago de Chile, Singapur y Viena con el objetivo de analizar la demanda de las diversas formas de transporte que existen en esas ciudades y establecer políticas adecuadas.
      * __*Machine Learning*__ -> Carpeta que contiene documentación de apoyo sobre el aprendizajeautomático.
       * __*machine_learning_a_bayesian_perspective.pdf*__ -> Documento sobre información de aprendizaje automático en general en general.
       * __*machine_learning_mastery_with_python.pdf*__ -> Documento que contiene información acerca del aprendizaje automático en Python.
       * __*tips, tricks and hacks that you can use to make better predictions.pdf*__ -> Documento que contiene consejos para realizar mejores predicciones en Python.  
        automático.
      * __*PostgreSQL*__ -> Carpeta que contiene documentación de apoyo sobre PostgreSQL.
        * __*PostgreSQL.pdf*__ -> Documento que contiene información sobre PostgreSQL.
      * __*Python*__ -> Carpeta que contiene documentación de apoyo sobre Python.
        * __*CheatSheets*__ -> Carpeta que contiene documentos que resumen características que tiene el lenguaje Python para aprendizaje automático.
         * __*data-visualisation-infographics1.jpg*__ -> Imagen que contiene los distintos tipos de visualización de datos que posee Python.
         * __*LearnDataScienceInPython.jpg*__ -> Imagen que contiene de forma resumida los pasos a seguir para aprender ciencia de datos en Python.
         * __*Pandas.png*__ -> Imagen que contiene algunas características que ofrece la librería _Pandas_ de Python.
         * __*PythonForDataScience.pdf*__ -> Documento que contiene los principios básicos de Python para realizar ciencia de datos.
         * __*Scikit-Learn.pdf*__ -> Documento que contiene información acerca de los distintos algoritmos de aprendizaje automático que ofrece la librería _Scikit-Learn_ de Python.
       * __*CódigosEjemplo*__ -> Carpeta que contiene scripts de Python que muestran pequeños ejemplos de las carecterísticas de este lenguaje para ciencia de datos.
         * __*CorrelationsBetweenAttributes.py*__ -> Script de Python que muestra cómo averiguar la correlación entre determinados atributos.
         * __*datasets*__ -> Conjuntos de datos utilizados en los ejemplos.
           * __*pima-indians-diabetes.data.csv*__ -> Conjunto de datos utilizado para los ejemplos.
           * __*ima-indians-diabetes.names*__ -> Información acerca del conjunto de datos utilizado para los ejemplos.
         * __*DataTypeForEachAttribute.py*__ -> Script the Python que muestra el tipo de dato que tiene cada uno de los atributos cargados desde un fichero CSV.
         * __*DescriptiveStatistics.py*__ -> Script de Python que muestra estadísticos descriptivos de los datos cargados desde un fichero CSV.
         * __*DimensionsOfYourData.py*__ -> Script de Python que muestra las dimensiones de los datos cargados desde un fichero CSV
         * __*LoadCSVFiles_with_NumPy.py*__ -> Script de Python que muestra cómo cargar ficheros CSV con la biblioteca NumPy.
         * __*LoadCSVFiles_with_Pandas.py*__ -> Script de Python que muestra cómo cargar ficheros CSV con la biblioteca Pandas.
         * __*LoadCSVFiles_with_the_PythonStandardLibrary.py*__ -> Script de Python que muestra cómo cargar ficheros CSV con la biblioteca estándar de Python.
         * __*LoadCSV_from_URL_using_NumPy.py*__ -> Script de Python que muestra cómo cargar ficheros CSV con una URL utilizando la biblioteca NumPy.
         * __*LoadCSV_using_Pandas_from_URL.py*__ -> Script de Python que muestra cómo cargar ficheros CSV con una URL utilizando la biblioteca Pandas.
         * __*PeekAtYourData.py*__ -> Script de Python que muestra una pequeña parte de los datos cargados desde un fichero CSV.
       * __*python_pandas_tutorial.pdf*__ -> Documento extenso sobre la biblioteca Pandas.
       * __*ScipyLectures-simple.pdf*__ -> Documento extenso sobre la biblioteca Scipy.
 * **documentacionProyecto** -> Carpeta que contiene la documentación del proyecto del TFG.
    * __*arbolDirectorios.md*__ -> Documento Markdown que contiene el árbol de directorios del repositorio donde se aloja el proyecto del TFG, así como información de cada uno de los componentes del mismo.
       documentacionProyecto
       * __*Información KDDCup 2017*__ -> Carpeta que contiene la información de la competición KDDCup 2017.
        * __*Informacion KDDCup 2017.pdf*__ -> Documento que contiene la información proporcionada por la competición KDDCup 2017.
        * __*Route_KDD.pdf*__ -> Documento que contiene un esquema de las rutas de la competición junto con los enlaces que las forman.
       * __*Machine Learning*__ -> Carpeta que contiene la documentación del aprendizaje automático aplicado al proyecto del TFG.
        * __*Machine Learning del proyecto.pdf*__ -> Documento que contiene la información del aprendizaje automático del proyecto del TFG.
        * __*Machine Learning Models.pdf*__ -> Documento que contiene información sobre los distintos algoritmos de aprendizaje automático utilizados en el proyecto del TFG.
        * __*modelo-arima.pdf*__ -> Documento que contiene información sobre el modelo de predicción de series temporales denominado _ARIMA_.
       * __*PostgreSQL*__ -> Carpeta que contiene la documentación de las características de PostgreSQL utilizadas en el proyecto del TFG.
        * __*Carga de la base de datos con PostgreSQL.pdf*__ -> Documento que contiene la información sobre la carga de la base de datos del proyecto del TFG.
        * __*Manual PostgreSQL.pdf*__ -> Documento que contiene la información sobre los tipos de datos y los comandos de PostgreSQL utilizados en la carga de la base de datos del proyecto del TFG.
       * __*PresentacionesPDFMejoresResultados*__ -> Carpeta que contiene las presentaciones de los ganadores de la competición KDDCup 2017.
        * __*Travel Time Prediction*__ -> Carpeta que contiene las presentaciones de los ganadores de la competición KDDCup 2017 con respecto a la tarea de predicción del tiempo promedio de viaje.
          * __*Task1_1stPlace.pdf*__ -> Presentación del equipo que quedó en 1er puesto.
          * __*Task1_2ndPlace.pdf*__ -> Presentación del equipo que quedó en 2º puesto.
          * __*Task1_3rdPlace.pdf*__ -> Presentación del equipo que quedó en 3er puesto.
        * __*Volume Prediction*__ -> Carpeta que contiene las presentaciones de los ganadores de la competición KDDCup 2017 con respecto a la tarea de predicción del volumen de tráfico.
          * __*Task2_1stPlace.pdf*__ -> Presentación del equipo que quedó en 1er puesto.
          * __*Task2_2ndPlace.pdf*__ -> Presentación del equipo que quedó en 2º puesto.
          * __*Task2_3rdPlace.pdf*__ -> Presentación del equipo que quedó en 3er puesto.
* **Dudas, sugerencias y cosas que arreglar.md** -> Documento Markdown en el que se contemplan dudas, sugerencias y cosas que arreglar del proyecto del TFG.
* **graficas** -> Carpeta que contiene las gráficas del proyecto del TFG.
  * **grafica_con_promedio_tiempomediodeviaje_ todoslosdias** -> Carpeta que contiene las gráficas del tiempo promedio de viaje de todos los días de entrenamiento en todas las rutas.
    * __*generacion_grafica.py*__ -> Script de Python que genera las gráficas.
    * __*graficas_pdf.pdf*__ -> Documento que contiene las gráficas.
  * **grafica_con_promedio_volumentrafico_todoslosdias** -> Carpeta que contiene las gráficas del volumen de tráfico de todos los días de entrenamiento en todas las barreras de peaje en la dirección de entrada y salida (las rutas que permitan la salida).
    * __*generacion_grafica.py*__ -> Script de Python que genera las gráficas.
    * __*graficas_pdf.pdf*__ -> Documento que contiene las gráficas.
  * **graficas_iniciales.py** -> Script de Python que genera una serie de gráficas iniciales sobre los datos.
  * **graficas_primera_aproximacion_tiempopromedioviaje** -> Carpeta que contiene gráficas relacionadas con la primera aproximación del tiempo promedio de viaje.
    * __*generacion_grafica.py*__ -> Script de Python que genera las gráficas.
    * __*graficas_pdf.pdf*__ -> Documento que contiene las gráficas.
  * **graficas_tiempomedioviaje_intersecciones_por_día** -> Carpeta que contiene gráficas del tiempo promedio de viaje por día de entrenamiento en cada una de las intersecciones por horas.
    * __*graficas_tiempomedioviaje_intersecciones_por_día.py*__ -> Script de Python que genera las gráficas.
    * __*graficas_pdf.pdf*__ -> Documento que contiene las gráficas.
  * **graficas_tiempo_meteorologico** -> Carpeta que contiene gráficas del tiempo meteorológico de los datos de entrenamiento por días y por horas.
    * __*graficas_tiempo_meteorologico.py*__ -> Script de Python que genera las gráficas.
    * __*graficas_pdf.pdf*__ -> Documento que contiene las gráficas.
  * **graficas_volumentrafico_intersecciones_por_dia** -> Carpeta que contiene gráficas del volumen de tráfico en cada uno de los días de entrenamiento por horas en cada una de las barreras de peaje en la dirección de entrada y salida (aquellas que la tengan).
    * __*graficas_volumentrafico_intersecciones_por_dia.py*__ -> Script de Python que genera las gráficas.
    * __*graficas_pdf.pdf*__ -> Documento que contiene las gráficas.
* **machineLearningProyecto** -> Carpeta que contiene todo lo relacionado con el aprendizaje automático aplicado al proyecto del TFG.
  * **Aproximaciones** -> Carpeta que contiene las distintas aproximaciónes de aprendizaje automático aplicado a los datos.
    * __*Tiempo promedio de viaje*__ -> Carpeta que contiene las distintas aproximaciones del tiempo promedio de viaje.
      * __*1ª aproximacion*__ -> Carpeta que contiene la primera aproximación de predicciones del tiempo promedio de viaje.
        * __*borrarTablas_interstollgtime_test1.sql*__ -> Script de Python que borra las vistas creadas por cada ruta-intervalo.
        * __*firstMachineLearningAproximation.py*__ -> Script de Python que realiza las predicciones de tiempo promedio de viaje de la primera aproximación.
        * __*firstMachineLearningAproximation_otherMAPE.py*__ -> Script de Python que realiza las mismas predicciones que el script anterior excepto en que calcula el error MAPE de forma diferente.
        * __*primera_aproximacion_tfgdatosmodificados.sql*__ -> Script de PostgreSQL que permite añadir unas columnas necesarias a una serie de tablas y generar las tablas necesarias sobre los datos de entrenamiento para poder realizar las predicciones.
        * __*primera_aproximacion_tfgtest1_construccion_tablaconintervalosdoshorasantes.sql*__ ->  Script de PostgreSQL que permite añadir unas columnas necesarias a la tabla que almacena el tiempo promedio de viaje de los intervalos de 2 horas antes de los intervalos (este script se realizó por si era necesario para las predicciones).
        * __*primera_aproximacion_tfgtest1.sql*__ -> Script de PostgreSQL que permite añadir unas columnas necesarias a una serie de tablas y generar las tablas necesarias sobre los datos de testeo para poder realizar las predicciones.
      * __*2ª aproximacion*__ -> Carpeta que contiene la segunda aproximación de predicciones del tiempo promedio de viaje.
        * __*secondMachineLearningApproach_ARIMA.py*__ -> Script de Python realiza las predicciones de la segunda aproximación de predicciones mediante el modelo ARIMA.
    * __*Volumen de tráfico*__ -> Carpeta que contiene las distintas aproximaciones del volumen de tráfico.
      * __*Primera aproximacion*__ -> Carpeta que contiene la primera aproximación de predicciones del volumen de tráfico.
        * __*nohaynada.txt*__ -> Archivo que indica que todavía no hay nada.
  * **codigosEjemploTecnicasMachineLearning** -> Carpeta que contiene ejemplos en Python del uso de distintos modelos de aprendizaje automático.
    * __* ARIMA*__ -> Carpeta que contiene ejemplos del modelo ARIMA en Python.
      * __* ARIMA_example.py*__ -> Script de Python que contiene un ejemplo de aplicación del modelo ARIMA sobre una serie de datos.
      * __* shampoo-sales.csv*__ -> Conjunto de datos utilizados para el ejemplo.
      * __* tuningParametersARIMA_example.py*__ -> Script de Python que contiene otro ejemplo de aplicación del modelo ARIMA sobre una serie de datos.
    * __*KNN*__ -> Carpeta que contiene ejemplos del modelo KNN en Python.
      * __* iris.data*__ -> Conjunto de datos utilizados para el ejemplo.
      * __*KNN_example.py*__ -> Script de Python que contiene un ejemplo de aplicación del modelo KNN sobre una serie de datos.
    * __*LightGBM*__ -> Carpeta que contiene ejemplos del modelo LightGBM en Python.
      * __* adult.csv*__ -> Conjunto de datos utilizados para el ejemplo.
      * __*XGBoostvsLightGBM.py*__ -> Script de Python que contiene un ejemplo de aplicación del modelo XGBoost comparado con el modelo LightGBM.
    * __*Multiple Layer Perception*__ -> Carpeta que contiene ejemplos del modelo de Redes Neuronales en Python.
      * __*wine_classification.py*__ -> Script de Python que contiene un ejemplo de aplicación del modelo de Redes Neuronales sobre una serie de datos.
      * __*wine_data.csv*__ -> Conjunto de datos utilizados en el ejemplo.
    * __*XGBoost*__ -> Carpeta que contiene ejemplos del modelo XGBoost en Python.
      * __*xgboost_example.py*__ -> Script de Python que contiene un ejemplo de aplicación del modelo XGBoost sobre una serie de datos.
      * __*pima-indians-diabetes.csv*__ -> Conjunto de datos utilizados en el ejemplo.
  * **Scripts de utilidad** -> Carpeta que contiene scripts de utilidad a aplicar sobre el aprendizaje automático del proyecto.
    * **borrarVistas.sql** -> Script de PostgreSQL que borra una serie de vistas creadas para las predicciones.
* **README.md** -> Documento de presentación del repositorio del proyecto del TFG.
* **Tareas.md** -> Documento que recoge las distintas tareas realizadas y pendientes en cada una de las reuniones realizadas.
