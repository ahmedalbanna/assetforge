# التقييم الفني والتحليل المعماري لتقنيات إعادة البناء ثلاثي الأبعاد الفورية: دراسة مقارنة لمشروعي DreamGaussian و TripoSR

تُمثل عملية توليد النماذج ثلاثية الأبعاد من صورة واحدة ثنائية الأبعاد قفزة نوعية في مجالات الرؤية الحاسوبية والذكاء الاصطناعي التوليدي، واضعةً حداً لعقود من النحت اليدوي المعقد والمكلف. تقليدياً، اعتمدت خطوط الإنتاج الصناعي على تقسيم التصميم إلى مراحل متتالية تبدأ ببناء المجسمات متوسطة التفاصيل (Mid-Poly) لتجسيد الهيكل العام، تليها مرحلة النحت الرقمي الدقيق لإنتاج مجسمات عالية التفاصيل (High-Poly)، وانتهاءً بتبسيط الشبكة المضلعة لاستخراج مجسمات منخفضة التفاصيل (Low-Poly) صالحة للرندرة الفورية مع تطبيق تقنيات تغليف وفرد إحداثيات الإكساء (UV Unwrapping) وبيكسل الألوان. ومع ظهور النماذج العصبية التوليدية، ركزت الجهود الأولى على الاستمثال القائم على حقول الإشعاع العصبي (NeRF) عبر أسلوب أخذ عينات تقطير الدرجات (Score Distillation Sampling - SDS). ورغم كفاءة هذه الطرق في تمثيل التفاصيل البصرية، إلا أنها عانت من بطء شديد في التقارب والتحسين لكل عينة على حدة، ما حد من تطبيقاتها العملية. في هذا السياق، برز مشروعا "DreamGaussian" و"TripoSR" كأدوات مفتوحة المصدر مرخصة تحت ترخيص "MIT" المرن، مقدمين حلولاً جذرية لتسريع التوليد ثلاثي الأبعاد إلى بضع ثوانٍ أو دقائق معدودة.   

## التحليل المعماري والهندسي لنظام DreamGaussian

يقدم مشروع DreamGaussian، الذي طوره الباحثون جياشيانغ تانغ، وجياوي رين، وهانغ تشو، وزيوي ليو، وعام زانغ، فلسفة تصميمية تعتمد على كسر معضلة الوقت في نماذج الاستمثال ثلاثي الأبعاد. وتكمن الفكرة الأساسية للمشروع في استبدال تمثيل الحجم العصبي الضمني بتمثيل صريح قائم على تشتت غاوس ثلاثي الأبعاد (3D Gaussian Splatting)، مقترناً بخوارزميات مخصصة لاستخراج الشبكة المضلعة (Mesh) وصقل الأكسية في المساحة الإحداثية ثنائية الأبعاد (UV Space).   

```
[صورة إدخال ثنائية الأبعاد] 
       │
       ▼ (إزالة الخلفية - Qin et al., 2020)
[صورة معزولة وممركزة]
       │
       ▼ 
[المرحلة الأولى: تحسين تشتت غاوس (3DGS) عبر خسارة SDS] (500 خطوة تدريبية)
       │
       ▼ 
[استعلام الكثافة المحلية واستخراج الشبكة المضلعة] (عبر خوارزمية Cignoni et al., 2008)
       │
       ▼ 
[فرد إحداثيات UV والإسقاط اللوني الخلفي] (Young, 2021)
       │
       ▼ 
[المرحلة الثانية: صقل الأنسجة في مساحة UV عبر خسارة MSE] (50 خطوة تدريبية)
       │
       ▼
[مجسم ثلاثي الأبعاد نهائي عالي الجودة]
```

### مرحلة تشتت غاوس التوليدي الاستمثالية

تبدأ العملية بمعالجة الصورة المدخلة عبر خوارزمية إزالة الخلفية وممركزة العنصر الأمامي لضمان جودة الاستمثال. تعتمد هندسة النظام على مرحلتين متتاليتين:   

في المرحلة الأولى، يتم تكييف تمثيل تشتت غاوس ثلاثي الأبعاد (3DGS) ليصبح إطار عمل توليدياً. تتم تهيئة الجسيمات الغاوسية بشكل عشوائي داخل كرة ذات نصف قطر محدد هندسياً بـ 0.5، وبقيمة شفافية (Opacity) أولية تبلغ 0.1 وبلون رمادي موحد. تبلغ هذه التهيئة 5000 جزيء للمهام المعتمدة على الصور (Image-to-3D) و1000 جزيء للمهام المعتمدة على النصوص (Text-to-3D). يمتد الاستمثال لـ 500 خطوة فقط، حيث يجري تكثيف وتقليم الجسيمات الغاوسية ديناميكياً كل 100 خطوة للصور (وكل 50 خطوة للنصوص)، مما يسرع تقارب الهيكل الأساسي مقارنة بتقليم الإشغال الحجمي البطيء في NeRF.   

تتصاعد دقة الرندرة أثناء التدريب من دقة منخفضة 64×64 بيكسل إلى دقة 512×512 بيكسل لتوليد تماسك تدريجي في التفاصيل. يتم أخذ عينات عشوائية لزوايا الكاميرا عند نصف قطر ثابت يبلغ 2 لمهام الصور و 2.5 لمهام النصوص، وبزاوية رؤية رأسية (y-axis FOV) تبلغ 49 درجة، مع تحديد زاوية السمت (Azimuth) في النطاق [−180,180] درجة وزاوية الارتفاع (Elevation) في النطاق [−30,30] درجة. كما يتم رندرة الخلفية عشوائياً باللون الأبيض أو الأسود لمساعدة الشبكة على فصل الكائن عن محيطه.   

تعتمد دالة خسارة هذه المرحلة على توجيه نماذج الانتشار ثنائية الأبعاد (مثل Zero-1-to-3 لمهام الصور و Stable Diffusion لمهام النصوص) لتطبيق خسارة أخذ عينات تقطير الدرجات (SDS). وتتكامل هذه الخسارة مع زيادة خطية لأوزان اللون ورندرة الشفافية الفراغية خلال عملية التدريب من القيمة صفر لتصل إلى قيمها النهائية البالغة 104 لوزن RGB و 103 لوزن الشفافية.   

### مرحلة استخراج الشبكة المضلعة والتحسين في مساحة UV

نظراً لأن مخرجات تشتت غاوس هي في الأساس نقاط غير مهيكلة ومبعثرة في الفضاء، فإن دمجها المباشر في محركات الرندرة التقليدية يواجه صعوبات هندسية بالغة. لحل هذه الأزمة، يقدم DreamGaussian خوارزمية لاستخراج شبكة مضلعة واضحة وسلسة عبر استعلام الكثافة المحلية (Local Density Querying) على شبكة حجمية بدقة معينة مع تطبيق حد كثافة تجريبي ثابت يساوي 1. يتم تلطيف وصقل هذه الشبكة المستخرجة هندسياً عبر خوارزميات التخفيض وإعادة التشبيك المتطورة لجعل المضلعات أكثر تماسكاً وتناسقاً.   

تتبع ذلك عملية إسقاط الألوان الخلفي (Color Back-projection) لتكوين خرائط الأنسجة. يتم فرد إحداثيات UV للمجسم المستخرج وتجهيز صورة إكساء فارغة. لملء هذه الخريطة اللونية، يتم رندرة الكائن هندسياً من 8 زوايا سمتية موحدة و3 مستويات ارتفاع، بالإضافة إلى المنظورين العلوي والسفلي، وبث هذه الألوان الملتفتة مجدداً فوق الأسطح.   

لتحسين جودة الأنسجة والتخلص من الضبابية الملازمة لخسارة SDS، يُطبق النظام المرحلة الثانية والمكونة من 50 خطوة تدريبية فقط لصقل النسيج التوليدي مباشرة في الفضاء ثنائي الأبعاد (UV Space). تستهدف هذه العملية رندرة المشهد عند زوايا عشوائية بدقة تتراوح من 128×128 إلى 1024×1024 بيكسل، وتطبيق نموذج انتشار مع تحديد وقت البداية لإضافة نويز خفيف tstart​ يحافظ على البنية الهيكلية ويسمح بإضافة تفاصيل بصرية دقيقة. تحسب الفروق البكسلية بواسطة دالة خسارة متوسط مربعات الخطأ (MSE):   

LMSE​=∥I^−Irefined​∥22​

تتيح هذه الهندسة الثنائية إنتاج مجسمات مضلعة ملونة بدقة تماثل الصور الحقيقية في غضون دقيقتين فقط على كرت شاشة فردي، محققة تسارعاً مذهلاً يقارب 10 أضعاف الأساليب التقليدية.   

## البنية المعمارية والتدريبية لنموذج TripoSR

يمثل مشروع TripoSR اتجاهاً موازياً في الفكر التصميمي لهندسة الأنظمة ثلاثية الأبعاد، حيث يرتكز بالكامل على الاستدلال المباشر والتغذية الأمامية (Feed-Forward) من لقطة واحدة دون الحاجة لأي عمليات استمثال تكرارية أثناء وقت التشغيل. يرتكز المشروع هندسياً على نموذج إعادة البناء الضخم (Large Reconstruction Model - LRM)، ويستطيع توليد المجسم بالكامل في زمن يقل عن 0.5 ثانية على بطاقة رسوميات متطورة.   

```
[صورة إدخال مفردة ثنائية الأبعاد]
       │
       ▼
[مشفّر الصور: Vision Transformer (DINOv1)] ──► [متجهات السمات الكامنة (Latent Vectors)]
                                                        │
                                                        ▼
[مفكك الترميز: Image-to-Triplane Decoder] ◄─────────────┘
  - انتباه ذاتي لسمات Triplane
  - انتباه متبادل لدمج السمات البصرية ثنائية الأبعاد
       │
       ▼
[التمثيل ثلاثي المستويات المحسن (Triplane, 40 قنوات)]
       │
       ▼
[حقل الإشعاع العصبي الحجمي (Triplane-NeRF MLP)]
       │
       ▼ (تطبيق خوارزمية Marching Cubes وخيار خبز الأنسجة)
[مجسم ثلاثي الأبعاد نهائي ملون أو مغلف بنسيج UV]
```

### المكونات المعمارية لـ TripoSR

تتكامل في البنية البرمجية لـ TripoSR ثلاثة مكونات رئيسية تعمل بانسجام تام:   

- **مشفّر الصور (Image Encoder):** يعتمد على نموذج محول الرؤية الفريد DINOv1 مسبق التدريب. تكمن كفاءة هذا المشفر في قدرته الفائقة على تمثيل الملامح الهندسية واستخلاص السمات المحلية والعالمية من رقع الصورة المدخلة وتحويلها إلى متجهات كامنة غنية بالمعلومات البنائية للهدف ثلاثي الأبعاد.   
    
- **مفكك ترميز الصورة إلى التمثيل ثلاثي المستويات (Image-to-Triplane Decoder):** يحول هذا المفكك متجهات الميزات الكامنة المستخلصة ثنائية الأبعاد إلى تمثيل ثلاثي المستويات متعامد وموجز (Triplane-NeRF). تكمن ميزة المستويات الثلاثة في قدرتها الاستثنائية على تمثيل الهياكل الهندسية المعقدة بفاعلية بصرية قصوى واستخدام منخفض لذاكرة الرام. يتكون مفكك الترميز من مكدس من طبقات المحولات (Transformers) التي تحتوي على آليات الانتباه الذاتي (Self-Attention) للربط المعرفي بين المستويات الثلاثة، وآليات الانتباه المتبادل (Cross-Attention) لجلب ودمج الميزات من متجهات الصورة الكامنة. تم استمثال وتحديد قنوات تمثيل المستويات الثلاثة بدقة عند 40 قناة للتوازن الأمثل بين تفاصيل المجسم والتحميل على كرت الشاشة.   
    
- **حقل الإشعاع العصبي المعتمد على المستويات الثلاثة (Triplane-based NeRF):** يتكون من شبكة عصبية متعددة الطبقات (MLP) مدمجة، تنحصر مسؤوليتها في قراءة إحداثيات النقطة من الفضاء وحساب قيمة الكثافة (Density) واللون المرتبطين بها لصياغة المجسم العصبي.   
    

### الابتكارات الرياضية والتدريبية للنموذج

يضم نموذج TripoSR عدة ابتكارات برمجية ساهمت في تفوقه النوعي:   

- **إلغاء تكييف الكاميرا الصريح:** بخلاف تصاميم LRM القياسية التي تفرض معرفة مسبقة ومثالية بوضعية الكاميرا، تم تصميم TripoSR ليتنبأ ويخمن مصفوفات الكاميرا الذاتية والخارجية تلقائياً أثناء التدريب والاستدلال، مما منحه مرونة تشغيلية عالية جداً للتعامل مع الصور العشوائية والواقعية المأخوذة في بيئات برية غير معالجة.   
    
- **التدريب المكثف على البيانات:** تم تدريب النموذج لمدة 5 أيام متواصلة باستخدام 22 عقدة حاسوبية فائقة، تضم كل منها 8 كروت شاشة من نوع NVIDIA A100 40GB. واستغل التدريب أسلوب رندرة متطور يحاكي التوزيع الضوئي والبصري الفعلي للعالم الحقيقي بالاعتماد على نسخ منتقاة من قاعدة بيانات Objaverse.   
    
- **دوال الخسارة والإشراف الرندري عالي الدقة:** يعتمد النموذج كلياً على الإشراف الرندري عبر مقارنة الصور المستخرجة مع الصور الحقيقية المرجعية لزوايا المراقبة. لإنهاء مشكلة العيوب الهندسية والجسيمات العائمة الطافية في الفراغ (Floaters)، تم دمج خسارة قناع ثنائية الانتروبيا المتبادلة (BCE Mask Loss):   
    

Lmask​(M^v​,MvGT​)=BCE(M^v​,MvGT​)

تتكامل هذه الخسارة مع خسائر رندرة الصور الأساسية كخسارة متوسط مربعات الخطأ (LMSE​) وخسارة الإدراك الحسي للأنسجة (LLPIPS​)، لتتألف المعادلة الكلية الحاكمة لعملية البناء والتدريب الفوري على النحو التالي:   

Lrecon​(I)=V1​v=1∑V​(LMSE​(I^v​,IvGT​)+λLPIPS​LLPIPS​(I^v​,IvGT​)+λmask​Lmask​(M^v​,MvGT​))

ويطبق النموذج استراتيجية انتقاء العينات ذات الأهمية (Importance Sampling) لضمان دقة التقاط ملامح الكائنات السطحية وتفاصيل الهياكل الميكروية دون استهلاك موارد المعالجة بلا نفع.   

## مقارنة تقنية ومعمارية شاملة

لتقديم صورة واضحة ومقاصد هندسية صريحة لمديري المشاريع ومطوري النظم ثلاثية الأبعاد، يقارن الجدولان التاليان الأداء التقني ومتطلبات التدريب والتشغيل للمشروعين استناداً إلى البيانات والنتائج المسجلة في أبحاثهما وتجاربهما العملية:

### المواصفات الفنية ومخرجات الأنظمة

|الخاصية الفنية|مشروع DreamGaussian|مشروع TripoSR|
|---|---|---|
|**المنهجية البرمجية وعقيدة النظام**|استمثال تصاعدي تكراري هجين قائم على تشتت غاوس (3DGS)|استدلال فوري ذو تغذية أمامية (Feed-Forward) مبني على المحولات|
|**التمثيل الفراغي الوسيط**|سحابة نقطية غاوسية صريحة ومكثفة تدريجياً|تمثيل ثلاثي المستويات الحجمي المتراص (Triplane-NeRF)|
|**زمن التوليد الفعلي**|دقيقتان تقريباً (دقيقة لكل مرحلة في مهام الصور)|أقل من 0.5 ثانية على بطاقة رسوميات A100 متطورة|
|**نمط الإكساء والتلوين القياسي**|نسيج مصقول فائق الدقة مبني ومعدل في مساحة إحداثيات UV|تلوين رؤوس المجسم (Vertex Colors) افتراضياً أو خبز نسيج UV بالطلب|
|**دقة خرائط الأنسجة المتوفرة**|دقة متغيرة مدعومة ترتفع حتى 1024 بيكسل وأكثر|دقة اختيارية قابلة للتعديل والخبز حتى نطاق 2K PBR|
|**كثافة المضلعات المستخرجة**|مرنة ومحسنة بخوارزميات إعادة التشبيك التلقائي|تصل افتراضياً حتى 100K مضلع وتتجاوز 2M في تحديثات v3.0|
|**رخصة المشروع مفتوح المصدر**|رخصة MIT البرمجية الحرة والمفتوحة|رخصة MIT البرمجية المفتوحة بالكامل|

  

### متطلبات الاستدلال والتدريب والبيئات التشغيلية

|المتطلب التقني|مشروع DreamGaussian|مشروع TripoSR|
|---|---|---|
|**الحد الأدنى لذاكرة كرت الشاشة (VRAM)**|6 إلى 12 جيجابايت للاستدلال؛ وتتطلب 24 جيجابايت لواجهات الرسوميات|6 جيجابايت فقط للاستدلال القياسي؛ و 8 جيجابايت للإنتاج الفعلي|
|**العتاد الرسومي الأدنى الموصى به**|كروت GeForce RTX 3070 أو V100 مع دعم CUDA 11.6 فما فوق|كروت RTX 3050 أو RTX 3060 فما فوق مع توافق CUDA 11.8|
|**البيئة التدريبية وبنية التطوير الفوقية**|تحسين محلي مستمر بخسارة SDS بالاعتماد على Threestudio و Diffusers|تدريب ضخم لـ 5 أيام على 22 عقدة تضم 176 كرت شاشة A100|
|**مجموعة بيانات التدريب المرجعية**|لا تتطلب قاعدة بيانات تدريب للنموذج نفسه؛ تستغل معرفة نماذج الانتشار|تدريب كامل مسبق على عينات منقاة من قاعدة بيانات Objaverse|
|**التوافق وأنظمة التشغيل المدعومة**|تشغيل ممتاز على أنظمة Linux Ubuntu و Windows مع مكاتب PyTorch و Cuda|دعم فائق لجميع المنصات المحلية والسحابية ولأجهزة MacOS (Apple Silicon)|

  

### التموضع التنافسي بين النماذج البديلة

عند مقارنة المشروعين بالنماذج التوليدية المتطورة في السوق، تتضح الأنماط والمقايضات التصميمية بدقة؛ فنماذج مثل "SF3D" الصادرة حديثاً تهدف لخدمة توليد متعدد الرؤى خلال 2.3 ثانية مستهلكة ذاكرة VRAM تصل لـ 9 جيجابايت، موفرة اتساقاً رائعاً للسطوح والأكسية. بينما تقدم نماذج عائلات "CRM" كفاءة هندسية عالية عبر رندرة ستة مساقط متعامدة بصرية والتنبؤ بالمجسم في 10 ثوانٍ. وتتصدر النماذج فائقة الضخامة مثل "TRELLIS" جودة الأصول الرقمية بمواد PBR متكاملة غير أنها تضع قيوداً صارمة تتطلب كروت شاشة بذاكرة VRAM لا تقل عن 24 جيجابايت وزمن معالجة أطول.   

## التكامل البرمجي وخطوط الإنتاج الصناعي

إن القيمة الحقيقية لهذه المشاريع مفتوحة المصدر لا تكمن في الأبحاث الرياضية وحدها، بل في قابليتها العالية للتطبيق داخل برامج التصميم الهندسي والتوليدي المعتمدة تجارياً.   

### خطوط الإنتاج المعتمدة على محرك ComfyUI

يعد تكامل نموذج TripoSR داخل محرك المعالجة البصرية التوليدي الشهير ComfyUI، عبر حزم مخصصة مثل `ComfyUI-Flowty-TripoSR`، أحد أكثر الأنظمة تفضيلاً لمطوري أصول الألعاب والميتافيرس. يمكن للفنانين الرقميين تدوين سكربت برمي لإنشاء صورة شخصية أو عنصر بيئي بواسطة نموذج Stable Diffusion وثنائية الأبعاد، وتوجيه هذا المخرج فوراً عبر عقدة "TripoSR Generate" لترجمتها تلقائياً لملف مجسم ثلاثي الأبعاد بصيغة OBJ أو PLY في أجزاء من الثانية.   

تميز هذا التكامل بإطلاق تحديثات متلاحقة مثل خوارزمية تريبو 3.0 (Tripo v3.0). تقدم هذه النسخة ترقيات تقنية واعدة تشمل إنتاج تفاصيل هيكلية شديدة الدقة كالنقوش الميكروية والكتابات المحفورة بدقة عالية ومحاكاة فائقة الأداء لسلوك المواد الفيزيائية (PBR) واللون في البيئات الرندرية. يتضمن هذا الإصدار أيضاً نمطين للإنتاج:   

- **النمط القياسي (Standard Mode):** الملائم للتفاعل اللحظي ومصممي أصول واجهات الويب الرقمية.
    
- **النمط الفائق (Detailed/Ultra Mode):** القادر على تصعيد كثافة المجسم لأكثر من مليوني مضلع لحصد أدق التفاصيل الهندسية والجمالية للمنتجات عالية الدقة.   
    

### الإضافة البرمجية لبرنامج Blender وحل عقبة تلوين الرؤوس يدوياً

توفر إضافة TripoSR البرمجية لبرنامج التصميم الشهير Blender منصة عمل مثالية تمكن الفنانين من معالجة المجسم كشبكة أولية (Base Mesh) وليس كأصل نهائي جاهز. يستغل المصممون هذه المخرجات السريعة لتجاوز مرحلة النمذجة الأولية الطويلة، مستفيدين من أدوات Blender الأصلية للقيام بمهام إعادة الهيكلة السلكية المنتظمة (Retopology)، تلطيف السطوح، صقل الإضاءة، والتحجيم الفيزيائي المتناسق مع محرك الألعاب المستهدف.   

بيد أن عقبة تقنية بارزة واجهت المصممين في النسخة القياسية لـ TripoSR، حيث يخرج النموذج الألوان مشفرة بداخل ألوان رؤوس المجسم (Vertex Colors). يترتب على هذا الأسلوب ضياع كامل لبيانات الألوان والظلال عند قيام المصممين بتبسيط المجسم أو تغيير كثافة مضلعاته.   

لحل هذه المعضلة الهندسية، يستعمل المطورون أدوات مثل حزمة `TripoSR-Bake` المجتمعية. تعتمد هذه الحزمة على استراتيجية فصل ذكية تسير وفق الخطوات التالية:   

- استدعاء نص الاستدلال الأول `01-mesh.py` لاستخراج تقريب أولي للمجسم الهندسي وتصديره كملف OBJ، وفي الوقت نفسه حفظ كافة ميزات حقل الإشعاع العصبي الكامنة المرندرة بداخل ملف ثنائي مشفر (PKL).   
    
- قيام المصمم بإجراء عمليات النمذجة والتعديل والتبسيط اليدوي للشبكة المضلعة في برنامج Blender لتقليل الوزن الهيكلي للمجسم، شرط الحفاظ على الحدود الخارجية للفضاء المحيط بالمجسم دون تعديل شامل للتحجيم أو الدوران.   
    
- استدعاء نص المعالجة الثاني `02-texture.py` لتوليد خريطة إحداثيات UV للشبكة المعدلة يدوياً، ومن ثم خبز وبث بيانات ميزات ملف NeRF (المحفوظ بملف PKL) مباشرة على خريطة النسيج الجديدة ثنائية الأبعاد بدقة تصل إلى 1024 بيكسل أو أكثر. يضمن هذا الحل الحفاظ الكامل على دقة وجودة تلوين المجسم الأصلية والعميقة فوق مجسم ذكي ذو كثافة مضلعات منخفضة ومثالية للاستخدام البرمجي الفوري.   
    

## قيود الأنظمة الحالية وآفاق التطور المستقبلي

على الرغم من القفزات المعمارية الكبيرة التي حققها كلا المشروعين، لا تزال هناك عقبات جوهرية تفرضها البنية الهندسية الحالية وتستلزم أبحاثاً تكميلية دقيقة لحلها.   

### معضلة هندسة تشتت غاوس (3DGS) التقليدية

تكمن الفجوة الرئيسية لتمثيل تشتت غاوس التقليدي في كونه يعتمد على حبات ونواة غاوسية ذات امتداد لامتناهٍ وغير محدود فراغياً. لتجاوز هذه العقبة البرمجية وتوفير رندرة سريعة، تضطر الخوارزميات إلى فرض قيم تصفية وقطع قسرية قاسية (Hard-coded cut-off)، مما يفرز مشغولات بصرية مشوهة (Cut-off artifacts) وحواف غير منتظمة. علاوة على ذلك، تفشل هذه الجسيمات المنحنية بطبيعتها في تجسيد السطوح المستوية الهندسية الصارمة والزوايا الحادة، مما ينتج عنه مجسمات ذات مظهر رطب أو ضبابي في بعض تفاصيلها الدقيقة.   

من ناحية أخرى، يستغل تشتت غاوس التوابع التوافقية الكروية منخفضة الرتبة (Spherical Harmonics - SH) لتشفير الألوان المعتمدة على زاوية الرؤية. يعيب هذا الأسلوب الرياضي نمو عدد المعاملات والبارامترات بشكل حدودي متسارع مع الرتب العليا من SH. ونتيجة لذلك، تضطر هذه النماذج للاكتفاء بالرتب الدنيا فقط، مما يؤدي إلى فرز انتقالات لونية شديدة البساطة والنعومة تفشل في التقاط الوميض واللمعان (Specular Highlights) والظلال الدقيقة الناتجة عن تداخل الإضاءات الفراغية المعقدة.   

### عقبات الأبعاد ومعضلة جانوس المزدوجة

تواجه نماذج التوليد المعتمدة على التوجيه ثنائي الأبعاد (مثل DreamGaussian) معضلة بصرية شهيرة تدعى "معضلة جانوس" (The Janus Problem)، حيث تتكرر الأوجه أو تفاصيل الجسد الأمامية على الجوانب الخلفية للمجسم التوليدي ثلاثي الأبعاد بشكل مشوه وباعث على عدم الواقعية. يعود منشأ هذه الأزمة إلى الانحياز البصري الشديد لنماذج الانتشار ثنائية الأبعاد. فنماذج الانتشار مسبقة التدريب على الصور ثنائية الأبعاد تمتلك انحيازاً إحصائياً لرسم وتكوين الجوانب الأمامية للأشياء نظراً لطبيعة توزيع بيانات الويب. وعند استدعاء النموذج لتخمين الرؤى الجانبية أو الخلفية للاستمثال ثلاثي الأبعاد، يقع في فخ "هياكل الوهم المكاني" (Spatial perception hallucinations)، مطبقاً سمات الوجه الأمامي على الظهر.   

وتتفاقم هذه الأزمة كلما ازدادت المطالبات النصية تفصيلاً وتعقيداً، حيث يقل اهتمام وتفسير النموذج للعبارات المكانية الموجهة لزوايا الكاميرا. لحل هذه العقدة المعمارية، تتبنى النظم الحديثة منهجيات هجينة مثل دمج تقنيات التوجيه السلبي المتعامد في فضاء الدرجات (Perp-Neg) للتخلص من الملامح الأمامية في زوايا الرؤية الخلفية، أو الاعتماد على بنيات دمج المعرفة الفراغية المباشرة ثلاثية الأبعاد كمشروع "DIRECT-3D".   

### المنظور المستقبلي والحلول الهندسية المتطورة

يتحرك قطاع الأبحاث نحو دمج التطورات الهندسية لإنتاج أصول ميكانيكية وفيزيائية دقيقة ومستقرة، وتتجسد هذه الجهود في ثلاثة محاور رئيسية:

- **تطبيقات الحركة والفيزياء المتكاملة:** برز مشروع "PhysGaussian" الذي يدمج ديناميكيات نيوتن الفيزيائية بداخل تمثيل تشتت غاوس مباشرة دون الحاجة لبناء شبكات مضلعة معقدة أو استخدام خوارزميات مكعبات السير. يمنح هذا الأسلوب الأصول الرقمية قدرة ذاتية على التفاعل الميكانيكي ومحاكاة التشوه والإجهادات الفيزيائية تزامناً مع الرندرة البصرية الفورية.   
    
- **أدوات تحسين الهندسة السطحية الموجهة:** يُطور مجتمع المصادر المفتوحة خوارزميات مثل "SuGaR" التي تفرض قيود تنظيم دقيقة لإجبار جسيمات غاوس على التراص والمحاذاة المثالية مع السطوح الهندسية الفعلية للمجسم. يتيح هذا الاتجاه استخراج شبكات مضلعة بالغة الدقة والتفاصيل عن طريق تطبيق خوارزمية بواسون (Poisson Reconstruction) السريعة والذكية كبديل رائد لعيوب المكعبات التقليدية.   
    
- **إعادة البناء ثنائي المرحلة المعتمد على الأسطح المتمايزة:** يهدف دمج آليات استخراج الأسطح المتمايزة القابلة للاشتقاق مباشرة بداخل طبقات المحول لنموذج الـ LRM (كما في مشروع MeshLRM) إلى تمكين النموذج التنبئي من إخراج الشبكة المضلعة المكسوة بالكامل بشكل هندسي متسق وخالٍ من الجسيمات الطافية، متجاوزاً بذلك عيوب الترجمة الضمنية لحقول NeRF الكلاسيكية.   
    

تبرهن هذه المشاريع الهندسية التوليدية الفورية أن دمج خوارزميات التمثيل الفراغي المرن مع تسريع التغذية الأمامية وتكامل شبكات المحولات يمهد لثورة غير مسبوقة تضع أدوات إنتاج المحتوى الاحترافي ثلاثي الأبعاد في متناول الفنانين والمطورين على حد سواء بمرونة مطلقة وسرعة تناهز اللحظية.   

[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2503.09439)

arxiv.org

SuperCarver: Texture-Consistent 3D Geometry Super-Resolution for High-Fidelity Surface Detail Generation - arXiv

Opens in a new window](https://arxiv.org/html/2503.09439)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fhuggingface.co%2Fpapers%2F2309.16653)

huggingface.co

Paper page - DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation - Hugging Face

Opens in a new window](https://huggingface.co/papers/2309.16653)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2F%40yousra.aoudi%2Fdreamgaussian-a-revolution-in-efficient-3d-content-creation-4d2b9d739b1f)

medium.com

DreamGaussian: A Revolution in Efficient 3D Content Creation | by Yousra Aoudi | Medium

Opens in a new window](https://medium.com/@yousra.aoudi/dreamgaussian-a-revolution-in-efficient-3d-content-creation-4d2b9d739b1f)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fdreamgaussian%2Fdreamgaussian.github.io)

github.com

GitHub - dreamgaussian/dreamgaussian.github.io: Project Page for Generative Gaussian Splatting for Efficient 3D Content Creation

Opens in a new window](https://github.com/dreamgaussian/dreamgaussian.github.io)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FVAST-AI-Research%2FTripoSR%2Fblob%2Fmain%2FLICENSE)

github.com

MIT License - VAST-AI-Research/TripoSR - GitHub

Opens in a new window](https://github.com/VAST-AI-Research/TripoSR/blob/main/LICENSE)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fproceedings.iclr.cc%2Fpaper_files%2Fpaper%2F2024%2Ffile%2F905202e21386913d8eac637c2b50f590-Paper-Conference.pdf)

proceedings.iclr.cc

DREAMGAUSSIAN: GENERATIVE GAUSSIAN SPLAT- TING FOR EFFICIENT 3D CONTENT CREATION - ICLR Proceedings

Opens in a new window](https://proceedings.iclr.cc/paper_files/paper/2024/file/905202e21386913d8eac637c2b50f590-Paper-Conference.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fstatic1.squarespace.com%2Fstatic%2F6213c340453c3f502425776e%2Ft%2F65e5d4991c13be738cc1f8fe%2F1728899860281%2FTripoSR_report.pdf)

static1.squarespace.com

TripoSR: Fast 3D Object Reconstruction from a Single Image - Squarespace

Opens in a new window](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65e5d4991c13be738cc1f8fe/1728899860281/TripoSR_report.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fabs%2F2309.16653)

arxiv.org

DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation - arXiv

Opens in a new window](https://arxiv.org/abs/2309.16653)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2309.16653v1)

arxiv.org

DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation - arXiv

Opens in a new window](https://arxiv.org/html/2309.16653v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Ficcvm.org%2F2026%2Ffiles%2Fpapers%2F639.pdf)

iccvm.org

SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization - ICCVM

Opens in a new window](https://iccvm.org/2026/files/papers/639.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F396620577_A_three-tier_generative_AI_workflow_for_metaverse_asset_creation)

researchgate.net

(PDF) A three-tier generative AI workflow for metaverse asset creation - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/396620577_A_three-tier_generative_AI_workflow_for_metaverse_asset_creation)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3D6zROYoHlcp)

openreview.net

DiffGS: Functional Gaussian Splatting Diffusion - OpenReview

Opens in a new window](https://openreview.net/forum?id=6zROYoHlcp)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FVAST-AI-Research%2FTripoSR)

github.com

GitHub - VAST-AI-Research/TripoSR: TripoSR: Fast 3D Object Reconstruction from a Single Image

Opens in a new window](https://github.com/VAST-AI-Research/TripoSR)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2403.02151v1)

arxiv.org

TripoSR: Fast 3D Object Reconstruction from a Single Image - arXiv

Opens in a new window](https://arxiv.org/html/2403.02151v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.mdpi.com%2F2076-3417%2F16%2F6%2F2678)

mdpi.com

From 2D to 3D: A Generative Model from Single Image to Digital 3D of Chinese Three Gorges Cultural Relics - MDPI

Opens in a new window](https://www.mdpi.com/2076-3417/16/6/2678)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Ftrellis2.app%2Fblog%2Fbest-image-to-3d-models-huggingface)

trellis2.app

Best Image to 3D Models on HuggingFace (2026)

Opens in a new window](https://trellis2.app/blog/best-image-to-3d-models-huggingface)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fhuggingface.co%2Fstabilityai%2FTripoSR)

huggingface.co

stabilityai/TripoSR - Hugging Face

Opens in a new window](https://huggingface.co/stabilityai/TripoSR)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fiffyloop%2FTripoSR-Bake)

github.com

Generate 3D meshes from a single 2D image using TripoSR, complete with manual geometry editing and texture baking support - GitHub

Opens in a new window](https://github.com/iffyloop/TripoSR-Bake)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.triposrai.com%2Fposts%2Ftriposr-vs-sf3d-comparison-2025%2F)

triposrai.com

TripoSR vs SF3D: Which Image-to-3D Tool Is Better?

Opens in a new window](https://www.triposrai.com/posts/triposr-vs-sf3d-comparison-2025/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.triposrai.com%2Fposts%2Ftriposr-blender-addon-workflow-tutorial%2F)

triposrai.com

TripoSR Blender Add-on: Install & Workflow Guide

Opens in a new window](https://www.triposrai.com/posts/triposr-blender-addon-workflow-tutorial/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fblog.comfy.org%2Fp%2Fgenerate-high-fidelity-3d-models)

blog.comfy.org

Generate High-Fidelity 3D Models with Tripo v3.0 in ComfyUI

Opens in a new window](https://blog.comfy.org/p/generate-high-fidelity-3d-models)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fdreamgaussian%2Fdreamgaussian)

github.com

dreamgaussian/dreamgaussian: [ICLR 2024 Oral] Generative Gaussian Splatting for Efficient 3D Content Creation - GitHub

Opens in a new window](https://github.com/dreamgaussian/dreamgaussian)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FLakonik%2FMVEdit)

github.com

Lakonik/MVEdit: 3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation - GitHub

Opens in a new window](https://github.com/Lakonik/MVEdit)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.triposrai.com%2Fposts%2Fopen-source-3d-reconstruction-showdown%2F)

triposrai.com

Best Open Source 3D Reconstruction Tools Compared - TripoSR AI

Opens in a new window](https://www.triposrai.com/posts/open-source-3d-reconstruction-showdown/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fbeta.pinokio.co%2Fapps%2Fgithub-com-cocktailpeanutlabs-triposr)

beta.pinokio.co

TripoSR on Pinokio

Opens in a new window](https://beta.pinokio.co/apps/github-com-cocktailpeanutlabs-triposr)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.triposrai.com%2Fposts%2Ftriposr-comfyui-node-guide%2F)

triposrai.com

TripoSR ComfyUI Node: Install & Workflow Guide

Opens in a new window](https://www.triposrai.com/posts/triposr-comfyui-node-guide/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2F3DTopia%2FGeneMAN)

github.com

[NeurIPS 2025] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data - GitHub

Opens in a new window](https://github.com/3DTopia/GeneMAN)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.semanticscholar.org%2Fpaper%2FTripoSR%253A-Fast-3D-Object-Reconstruction-from-a-Image-Tochilkin-Pankratz%2F4299b79ef41601cf6e3e0603f7216d72b6d1315f)

semanticscholar.org

[PDF] TripoSR: Fast 3D Object Reconstruction from a Single Image - Semantic Scholar

Opens in a new window](https://www.semanticscholar.org/paper/TripoSR%3A-Fast-3D-Object-Reconstruction-from-a-Image-Tochilkin-Pankratz/4299b79ef41601cf6e3e0603f7216d72b6d1315f)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fflowtyone%2FComfyUI-Flowty-TripoSR)

github.com

flowtyone/ComfyUI-Flowty-TripoSR - GitHub

Opens in a new window](https://github.com/flowtyone/ComfyUI-Flowty-TripoSR)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2501.18630v2)

arxiv.org

Deformable Beta Splatting - arXiv

Opens in a new window](https://arxiv.org/html/2501.18630v2)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FICCV2025W%2FgDT-IV%2Fpapers%2FZhang_Improving_Viewpoint_Consistency_in_3D_Generation_via_Structure_Feature_and_ICCVW_2025_paper.pdf)

openaccess.thecvf.com

Improving Viewpoint Consistency in 3D Generation via Structure Feature and CLIP Guidance - CVF Open Access

Opens in a new window](https://openaccess.thecvf.com/content/ICCV2025W/gDT-IV/papers/Zhang_Improving_Viewpoint_Consistency_in_3D_Generation_via_Structure_Feature_and_ICCVW_2025_paper.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2409.06620v1)

arxiv.org

MVGaussian: High-Fidelity text-to-3D Content Generation with Multi-View Guidance and Surface Densification - arXiv

Opens in a new window](https://arxiv.org/html/2409.06620v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.ecva.net%2Fpapers%2Feccv_2024%2Fpapers_ECCV%2Fpapers%2F03300.pdf)

ecva.net

FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally - ECVA

Opens in a new window](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03300.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fneurips.cc%2Fvirtual%2F2024%2Fposter%2F93528)

neurips.cc

NeurIPS Poster Hallo3D: Multi-Modal Hallucination Detection and Mitigation for Consistent 3D Content Generation

Opens in a new window](https://neurips.cc/virtual/2024/poster/93528)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2412.02287v3)

arxiv.org

Viewpoint Consistency in 3D Generation via Training-free Attention and CLIP Guidance

Opens in a new window](https://arxiv.org/html/2412.02287v3)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fdreamfusion)

emergentmind.com

DreamFusion: Text-to-3D Synthesis - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/dreamfusion)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Ftamilselvanarjun%2Fawesome-3d-gaussian-splatting)

github.com

tamilselvanarjun/awesome-3d-gaussian-splatting - GitHub

Opens in a new window](https://github.com/tamilselvanarjun/awesome-3d-gaussian-splatting)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenreview.net%2Fpdf%3Fid%3DR1rNN22IoP)

openreview.net

MESHLRM: LARGE RECONSTRUCTION MODEL FOR HIGH-QUALITY MESHES - OpenReview

Opens in a new window](https://openreview.net/pdf?id=R1rNN22IoP)

[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/abs/2403.02151)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2406.07648v1)[

![](https://t3.gstatic.com/faviconV2?url=https://www.triposrai.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.triposrai.com/)[

![](https://t1.gstatic.com/faviconV2?url=https://papercopilot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://papercopilot.com/cat/cs/ai/aigc/text-to-3d/)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2309.16653v2)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2312.17142v3)[

![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://huggingface.co/papers?q=2D-to-3D%20lifter)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2410.09592v1)[

![](https://t3.gstatic.com/faviconV2?url=https://www.scitepress.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.scitepress.org/Papers/2025/133495/133495.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://www.tripo3d.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.tripo3d.ai/blog/explore/best-local-3d-model-ai-tools-guide)[

![](https://t0.gstatic.com/faviconV2?url=https://runtime-revolution.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://runtime-revolution.com/insights/running-deep-learning-models-as-applications-with-fastapi)[

![](https://t1.gstatic.com/faviconV2?url=https://www.tripo3d.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.tripo3d.ai/blog/collect/the-ultimate-guide-to-computer-specs-for--d-printing-15uaerisryi)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2602.03207v1)[

![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wu_100Editor_100_Views_per_Batch_and_Minute-Scale_View-Consistent_3D_Editing_CVPRF_2026_paper.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://huggingface.co/papers?q=Gaussian-weighted%20splicing)[

![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.researchgate.net/publication/391013521_A_review_on_3D_Gaussian_splatting_for_sparse_view_reconstruction)[

![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openaccess.thecvf.com/content/ICCV2025/papers/Han_REPARO_Compositional_3D_Assets_Generation_with_Differentiable_3D_Layout_Alignment_ICCV_2025_paper.pdf)[

![](https://t0.gstatic.com/faviconV2?url=https://ojs.aaai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://ojs.aaai.org/index.php/AAAI/article/view/37707/41669)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2412.01583v3)[

![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.mdpi.com/2076-3417/16/8/4026)[

![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openaccess.thecvf.com/content/WACV2026/papers/Yan_3DSceneEditor_Controllable_3D_Scene_Editing_with_Gaussian_Splatting_WACV_2026_paper.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2505.07747v1)[

![](https://t1.gstatic.com/faviconV2?url=https://www.tripo3d.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.tripo3d.ai/de/blog/explore/model-world-com)[

![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://huggingface.co/spaces/AP123/dreamgaussian)[

![](https://t0.gstatic.com/faviconV2?url=https://dreamgaussian.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dreamgaussian.github.io/)[

![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://huggingface.co/spaces/jiawei011/dreamgaussian)[

![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.scribd.com/document/819520546/DreamGaussian-a-Hugging-Face-Space-by-jiawei011)[

![](https://t0.gstatic.com/faviconV2?url=https://note.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://note.com/npaka/n/n0f04f3e69af6?hl=en)[

![](https://t0.gstatic.com/faviconV2?url=https://dev.to/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dev.to/ajmal_hasan/using-hugging-face-models-in-google-colab-a-beginners-35ll)[

![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://huggingface.co/spaces/stabilityai/TripoSR)[

![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://huggingface.co/spaces/AleenDG/3DGenTripoSR)[

![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://medium.com/google-colab/hugging-face-notebooks-x-colab-722d91e05e7c)[

![](https://t1.gstatic.com/faviconV2?url=https://colab.research.google.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://colab.research.google.com/github/google-health/medasr/blob/main/notebooks/quick_start_with_hugging_face.ipynb)[

![](https://t3.gstatic.com/faviconV2?url=https://www.kaggle.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.kaggle.com/datasets/agukcu/comfyui-flowty-triposr)[

![](https://t0.gstatic.com/faviconV2?url=https://www.runcomfy.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.runcomfy.com/comfyui-nodes/ComfyUI-Flowty-TripoSR)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Flowty-TripoSR-ZHO)[

![](https://t1.gstatic.com/faviconV2?url=https://fal.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://fal.ai/models/fal-ai/triposr)[

![](https://t1.gstatic.com/faviconV2?url=https://blenderartists.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://blenderartists.org/t/new-image-to-3d-blender-add-on/1519927)[

![](https://t3.gstatic.com/faviconV2?url=https://ai4culture.eu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://ai4culture.eu/resources/tools/21)