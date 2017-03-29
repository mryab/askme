# Ask me
## Рябинин Максим Константинович, группа 154

### 1. Цель проектной работы
  
  Изучить, воспроизвести и при возможности улучшить функционал нейронной сети, описанной в [статье](https://arxiv.org/pdf/1506.07285.pdf). В конечном итоге программный продукт будет представлять из себя приложение, принимающее некоторый контекст и релевантный ему вопрос на естественном языке и возвращающее ответ (также на естественном языке) на этот вопрос с учётом контекста.
  
### 2. Задачи проекта
  
  1. Изучить теорию нейронных сетей различных архитектур (в том числе рекуррентные) и методы их обучения
  2. Разобраться в методах, с помощью которых авторами статьи был достигнут описанный результат
  3. Построить модель, выполняющую соответствующую задачу
  4. Улучшить её результаты (быстродействие, качество)

### 3. Актуальность решаемой задачи

  В настоящее время глубокое обучение с использованием рекуррентных нейронных сетей является активно развивающейся отраслью, поэтому разработка нейронной сети, способной отвечать на задаваемые на естественном языке вопросы, представляет достаточно большой интерес как с точки зрения науки, так и из практических соображений. Вне всякого сомнения, решаемая задача является актуальной, так как конечный результат возможно будет применить на практике во множестве областей, в которых требуется взаимодействие с человеком, начиная с персонализированных помощников и заканчивая чат-ботами служб технической поддержки или интернет-магазинов.
  
### 4. Обзор существующих решений
  
  В указанной выше статье, изучение которой является важной частью данного проекта, описана структура выполняющей схожую задачу модели. Однако в ходе проекта также будет создана программная реализация, которую уже можно будет встроить в структуру масштабных сервисов.

### 5. Обзор используемых решений
  
  В качестве основного языка программирования используется Python, библиотеки глубокого обучения низкого уровня — TensorFlow и Theano, так как они обладают достаточно большим сообществом и позволяют ускорять вычисления с помощью видеокарты (так как в перспективе мы будем сравнивать возможности и быстродействие фреймворков, использоваться будут оба). В качестве библиотек глубокого обучения с более высоким уровнем абстракции используются Keras и, возможно, Lasagne, так как с их помощью возможно строить нейронные сети, менее вдаваясь в детали низкоуровневой реализации и уделяя больше времени архитектуре модели в целом.

### 6. План работы
  1. 19 марта — выполнены все лабораторные работы, начат разбор статьи и работа над основной частью проекта
  Лабораторные работы:
  - [x] Линейная бинарная классификация с использованием нескольких вариаций градиентного спуска (Adagrad, RMSprop, Adam, Adadelta, Nesterov Momentum).
  - [x] Многослойные полносвязные нейронные сети, их применение к многоклассовой классификации. Dropout, L1- и L2-регуляризация, автоэнкодеры.
  - [x] Использование свёрточных нейросетей для классификации изображений на примере рукописных иероглифов. (https://inclass.kaggle.com/c/ch-ch)
  - [x] Изучение принципов работы Word2Vec и GloVe (на текущий момент наиболее качественных способов векторного представления слов) и тренировка соответствующих моделей на текстах из Википедии.
  - [x] Применение рекурректных нейронных сетей для построения транскрипций слов английского языка (https://inclass.kaggle.com/c/en-phonetics)
  2. 16 апреля — завершено обсуждение статей
  3. 3 июня — построена и обучена финальная версия модели
  
