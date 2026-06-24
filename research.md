# الأساس الهندسي والرياضي لإعادة البناء ثلاثي الأبعاد: دراسة تحليلية متقدمة لمكتبتي OpenMVG و OpenMVS

## هندسة الرؤية الحاسوبية المتعددة المشاهد وسلسلة التصوير المساحي

تمثل عملية إعادة البناء ثلاثي الأبعاد من خلال مجموعة من الصور ثنائية الأبعاد المتداخلة الركيزة الأساسية للتصوير المساحي الرقمي الحديث وتطبيقات الرؤية الحاسوبية. ينقسم خط الأنابيب البرمجي لهذه العملية هندسياً إلى مرحلتين متتاليتين ومستقلتين رياضياً: مرحلة تقدير حركة الكاميرا والبنية الهندسية الهشة (Structure from Motion - SfM)، ومرحلة إعادة البناء الكثيف وإكساء الأسطح (Multi-View Stereo - MVS). تبرز في هذا النطاق مكتبتان برمجيتان مفتوحتا المصدر ومكتوبتان بلغة ++C تمثلان الأساس الرياضي الصارم لتنفيذ هذه المراحل، وهما مكتبة الرؤية المتعددة المشاهد المفتوحة (OpenMVG) ومكتبة الرؤية الاستريو متعددة المشاهد المفتوحة (OpenMVS).   

تأسست مكتبة OpenMVG كمنظومة برمجية متكاملة تهدف إلى حل مسألة البنية من الحركة (SfM)، وهي مستوحاة هندسياً من مكتبة `libmv` الشهيرة. تم تطويرها بتمويل ودعم أكاديمي وبحثي من مختبر LIGM-Imagine وشركات مثل Mikros Image و Foxel SA. صُممت المكتبة بالاعتماد على فلسفة هندسية صارمة تدعو إلى البساطة وقابلية الصيانة والتطوير الموجه بالاختبارات (Test-Driven Development)، وهي مرخصة بموجب رخصة Mozilla Public License 2.0 (MPL2). قام الباحثون بيير مولون (Pierre Moulon)، وباسكال موناس (Pascal Monasse)، وروموالد بيرو (Romuald Perrot)، ورينو مارليه (Renaud Marlet) بصياغة هذه المكتبة لتشكل بيئة رياضية مفتوحة تتيح للباحثين والمهندسين حلاً دقيقاً لمسألة المعايرة الخارجية للكاميرات واستخراج سحابة النقاط الهشة.   

في المقابل، صُممت مكتبة OpenMVS بواسطة مختبرات المطورين (cdcseacave) وبدعم علمي من بيير مولون لسد الفجوة الكبيرة التي كانت قائمة في البرمجيات المفتوحة المصدر المخصصة للمرحلة الأخيرة من سلسلة التصوير المساحي. فبينما كانت تتوفر مكتبات ناضجة لتقدير حركة الكاميرات وسحب النقاط الهشة، افتقرت الساحة البرمجية لمكتبات متكاملة وعالية الكفاءة قادرة على إعادة بناء الشبكات السطحية الكثيفة وإكسائها بنسيج واقعي. تقوم OpenMVS باستقبال مخرجات وضعيات الكاميرات وسحابة النقاط الهشة الناتجة عن OpenMVG أو COLMAP أو OpenSfM، وتقوم بتحويلها عبر سلسلة من الخوارزميات الحجمية والسطحية إلى مجسمات ثلاثية الأبعاد كاملة التفاصيل ومكسوة بأطلس نسيجي واقعي.   

## بنية مكتبة OpenMVG وسلسلة معالجة البنية من الحركة

تتميز مكتبة OpenMVG بقدرتها على العمل عبر منصات متعددة تشمل أنظمة Android و iOS و Linux و macOS (بما في ذلك معماريات Apple Silicon غير المعتمدة على x86) وأنظمة Windows. يتكون خط أنابيب SfM في المكتبة من سلسلة من الخطوات المعيارية التي يتم التحكم فيها عبر أدوات برمجية مخصصة:   

### 1. إدراج الصور والمعايرة الأولية (Image Listing)

تقوم أداة `openMVG_main_SfMInit_ImageListing` بفحص مجموعة الصور المدخلة وقراءة معطيات EXIF لتحديد البعد البؤري وحجم المستشعر، مما يسمح ببناء مصفوفة المعايرة الداخلية K للكاميرات. كما تتيح المكتبة أداة مخصصة تسمى `openMVG_main_SfMInit_ImageListingFromKnownPoses` لتهيئة المشاهد بناءً على وضعيات كاميرات معرفة مسبقاً من مجموعات البيانات القياسية مثل DTUMVS أو ETH3D.   

### 2. حساب واصفات الصور الفردية (Image Description)

تستدعي أداة `openMVG_main_ComputeFeatures` خوارزميات استخلاص المعالم البصرية. توفر المكتبة خيارات متعددة لاستخلاص الميزات تشمل ميزات SIFT الافتراضية المأخوذة من مكتبة VLFeat، وميزات AKAZE بنوعيها العائم (AKAZE_FLOAT) والثنائي (AKAZE_MLDB).   

تعتمد البنية الداخلية للمكتبة على الفصل الدقيق بين "الميزة" (Feature) التي تمثل الإحداثيات الهندسية للموقع ثنائي الأبعاد، وبين "الواصف" (Descriptor) الذي يمثل التمثيل الفوتومتري للمنطقة المحيطة بالنقطة. يتم تمثيل هذه المعالم بواسطة واصفات هندسية تسمى SIOPointFeature وهي اختصار لـ (Scale Invariant & Orientation)، حيث يتم تخزين الموقع الجغرافي ثنائي الأبعاد مقترناً بمقياس النقطة (s) واتجاهها (o) المتولد عن التدرج الفوتومتري.   

تتضمن هذه المرحلة آلية قوية للأقنعة (Masking)؛ حيث تبحث الأداة تلقائياً عن قناع عام باسم `global_mask.png` في مجلد الصور أو أقنعة فردية مخصصة لكل صورة بتنسيق `<image_name>_mask.png`. تمنع هذه الأقنعة الثنائية استخلاص الواصفات في المناطق السوداء (التي تمتلك القيمة صفر)، مما يسهم في التخلص من العناصر المتحركة في الخلفية أو تقييد البحث في مناطق هندسية محددة. كما تدعم الأداة استخلاص الميزات القائمة على الوضع الرأسي (Upright Features) عبر خيار `-u 1` لإلغاء خاصية الدوران الثابت عند الرغبة في تسريع المطابقة للصور الملتقطة رأسياً.   

### 3. مطابقة المعالم البصرية (Correspondence Matching)

تتولى أداة `openMVG_main_ComputeMatches` عملية البحث عن المتطابقات البصرية بين الصور. يتم حساب المسافات بين واصفات النقاط باستخدام طرق متعددة مثل مطابقة القوة الغاشمة (Brute-Force L2) أو أشجار FLANN المقترنة باختبار نسبة المسافة لديفيد لو (David Lowe) لمنع التطابقات العشوائية. تستعرض الأداة إحصائيات مفصلة لـ "مخطط الرؤية" (View Graph) لضمان متانة الربط بين الكاميرات.   

### 4. حل البنية والوضعيات الهندسية (SfM Solving)

توفر التحديثات الأخيرة لمكتبة OpenMVG أداة برمجية موحدة تتيح الوصول إلى ثلاثة محركات أساسية لحل مسألة SfM (التراكمي Incremental، التراكمي المطور IncrementalV2، والعالمي Global) من خلال واجهة تشغيل واحدة. يتم هنا استرجاع الوضعيات الخارجية للكاميرات وسحابة النقاط الهشة.   

يجب الانتباه هندسياً إلى أن سلسلة معالجة SfM تبدأ بفرضية توجيه المحور الصادي (Y-axis) نحو الأعلى، ولكن معالجة ضبط حزم الأشعة اللاحقة تمتلك الحرية الكاملة لتغيير اتجاهات المحاور. يؤدي هذا إلى اختلافات في اتفاقيات المحاور بين OpenMVG و OpenMVS؛ حيث تعتمد هندسة الرؤية الحاسوبية على المحور العيني (Z-axis) لتمثيل عمق الكاميرا، بينما تعتمد بيئات العرض ثلاثية الأبعاد الكلاسيكية على المحور العيني (Z-axis) كمحور رأسي للأعلى، وهو ما يتطلب معالجة دورانية عند التصدير للعوارض الخارجية.   

## الأساس الرياضي والخوارزمي المتطور لمكتبة OpenMVG

تستمد مكتبة OpenMVG دقتها الهندسية الفائقة من الصياغة الرياضية الصارمة لخوارزميات التقدير المتين والتحسين غير الخطي.   

### إطار التقدير المتين القائم على مبدأ التناقض (A Contrario RANSAC)

تواجه تطبيقات الرؤية الحاسوبية الكلاسيكية معضلة هندسية عند استخدام خوارزمية RANSAC التقليدية لتقدير مصفوفات التحويل الهندسي (مثل Fundamental أو Essential أو Homography Matrices)؛ حيث تتطلب هذه الخوارزميات تحديد عتبة مسافة ثابتة مسبقاً لفصل النقاط الصحيحة عن الشاذة. لكسر هذا القيد، صاغ مطورو OpenMVG خوارزمية AC-RANSAC الخالية من العتبات الثابتة.   

تعتمد هذه الخوارزمية على نظرية "النموذج العكسي" (A Contrario Framework)، حيث تُعرف الفرضية الصفرية (H0​) بأنها الحالة التي تتوزع فيها النقاط عشوائياً وبانتظام في المشهد دون أي ترابط هندسي حقيقي. يُعتبر النموذج الهندسي "معبراً وذا دلالة إحصائية" إذا كان احتمال ظهوره بمحض الصدفة تحت فرضية العدم ضئيلاً جداً. يُقاس هذا الترابط عبر دالة عدد الإنذارات الكاذبة (Number of False Alarms - NFA) للنموذج M عند امتلاك k نقطة مطابقة من أصل n نقطة، حيث تُرتب المسافات المتبقية للنقاط تصاعدياً، وتمثل Tk​ المسافة المتبقية للنقطة رقم k:   

NFA(M,k)=Ntests​(n,k,Nsample​)⋅P(residual≤Tk​∣M,H0​)k−Nsample​

حيث يعبر الحد Ntests​ عن العدد الكلي للعينات العشوائية المختبرة، ويمثل P(residual≤Tk​∣M,H0​) احتمال أن تمتلك نقطة عشوائية مسافة متبقية أقل من العتبة Tk​ تحت الفرضية الصفرية. نظراً لكون النقاط موزعة بانتظام عبر مساحة الصورة ذات الأبعاد w×h، فإن هذا الاحتمال يُعطى رياضياً بنسبة مساحة قرص نصف قطره Tk​ إلى مساحة الصورة الكلية:   

P(residual≤Tk​∣M,H0​)=w⋅hπTk2​​

تقوم خوارزمية AC-RANSAC بالبحث عن النموذج والحد k اللذين يقللان قيمة NFA إلى أقصى حد. إذا كانت القيمة الناتجة تحقق الشرط إحصائياً (log(NFA)<0)، يُقبل النموذج كحل هندسي موثوق، وتُعتبر القيمة Tk​ هي العتبة الديناميكية المثالية للفصل بين النقاط الصحيحة والشاذة لتلك الصورة المحددة دون أي تدخل خارجي.   

### الرياضيات الحاكمة لخط الأنابيب العالمي (Global SfM Pipeline)

يتميز خط الأنابيب العالمي في OpenMVG بقدرته على حل مسألة التوجيه الخارجي لجميع الكاميرات دفعة واحدة، ويتكون من الخطوات الرياضية المتتابعة التالية:   

1. **تقدير الدورانات النسبية المتينة:** يتم استخلاص مصفوفات الدوران النسبي Rij​ بين أزواج الكاميرات عبر تقدير المصفوفات الجوهرية (Essential Matrices) باستخدام خوارزمية AC-RANSAC والاعتماد على متجهات الاتجاه البصري (Bearing Vectors) للكاميرات المعايرة.   
    
2. **تصفية اتساق الدوران (Rotation Consistency):** تُزال الحواف الشاذة والخاطئة من مخطط الرؤية عبر تطبيق تصفية بايزية متسلسلة تعتمد على قياس وزن الدورانات المتراكمة على طول الحلقات المغلقة (Cycle Length Weighting) داخل المخطط.   
    
3. **حساب الدورانات المطلقة الكلية (Global Rotations):** تُحل مصفوفات الدوران المطلقة لجميع الكاميرات Ri​ بالتزامن باستخدام خوارزمية حل القيمة الذاتية المتفرقة (Sparse Eigenvalue Solver).   
    
4. **تخمين اتجاهات الترجمة النسبية عبر ثلاثيات الصور:** للتغلب على مشكلة غموض المقياس الهندسي ثنائي الأبعاد، يتم حساب اتجاهات الترجمة النسبية للمشاهد الثلاثية باستخدام ممتد ثلاثي البؤر المختزل (Reduced Trifocal Tensor) المدعوم بـ AC-RANSAC والتحسين المحدب لضمان السرعة والامتثال الهندسي.   
    
5. **حساب التراجم المطلقة الكلية (Global Translations):** يتم استنتاج مراكز الكاميرات المطلقة ci​ عبر حل نظام المعادلات غير الخطية الذي يربط بين اتجاهات التراجم النسبية ثنائية المشاهد tij​ ومراكز الكاميرات المطلقة:   
    

tij​=∥cj​−ci​∥cj​−ci​​

تُصاغ هذه المسألة باستخدام معيار اللانهائية (L∞​) لحل التراجم العالمية، وتدعم المكتبة ربطها بالمحلل الرياضي المتطور Mosek v6 LP للحصول على أداء وسرعة متفوقين في حل مسائل البرمجة الخطية الناتجة. كما يتضمن التحديث الأخير إضافة خوارزمية الترجمة العالمية الخطية (Linear Global Translation - LiGT) لتقدير التراجم المطلقة مباشرة من الدورانات العالمية وملاحظات تتبع النقاط.   

6. **ضبط حزم الأشعة الكلي (Global Bundle Adjustment):** تُجرى عملية تثليث نهائية لجميع المعالم لإنشاء سحابة النقاط الهشة، تليها عملية تحسين كلي غير خطي لمعلمات الكاميرات والنقاط معاً لتقليل خطأ إعادة الإسقاط البصري بالاستعانة بالمحلل المتطور Ceres Solver.   
    

## مكتبة OpenMVS: خط معالجة البيانات وإعادة البناء الكثيف

بمجرد اكتمال حساب البنية الهشة ووضعيات الكاميرات، يبدأ دور مكتبة OpenMVS لإنجاز المرحلة الكثيفة من عملية إعادة البناء. لا تعتمد OpenMVS برمجياً على OpenMVG بشكل مباشر، حيث تتوفر كمنظومة مستقلة، ولكنها تشتمل على واجهات ربط مرنة تتيح استقبال البيانات من خطوط معالجة متنوعة. تعتمد المكتبة في بنائها على حزمة متطورة من المكتبات مفتوحة المصدر تشمل Eigen (الإصدار 3.2 فما فوق)، OpenCV (الإصدار 2.4 فما فوق)، Ceres Solver (الإصدار 1.10 فما فوق)، CGAL (الإصدار 4.2 فما فوق)، Boost (الإصدار 1.56 فما فوق)، بالإضافة إلى مكتبتي VCGLib و GLFW الاختيارية لعرض وتداول الكائنات الهندسية.   

يتم تفعيل خط معالجة الكثافة في OpenMVS عبر الأدوات البرمجية الأربع التالية:   

### 1. توليد سحابة النقاط الكثيفة (DensifyPointCloud)

تعتبر هذه المرحلة جوهرية وحاسمة لتحديد دقة وجودة السطح النهائي. تستند هذه الأداة إلى خوارزمية مطابقة الرقع التصويرية العشوائية للرؤية الاستريو (PatchMatch Stereo) المأخوذة من الدراسة الأكاديمية الشهيرة لتقدير الأسطح الكثيفة للمشاهد الكبيرة. تتلخص آلية عملها في أربع خطوات أساسية متكررة:   

- **اختيار أزواج الصور (Stereo Pair Selection):** يتم تحديد أفضل صورة مرافقة (Target Image) لكل صورة مرجعية (Reference Image) بناءً على مدى مشاركتها للنقاط الهشة ومساحة التداخل البصري.   
    
- **التهيئة العشوائية (Random Initialization):** يُهيأ كل بكسل في الصورة المرجعية بقيمة عمق عشوائية على طول شعاع الرؤية البصري، مقترنة بمتجه عمودي (Normal vector) عشوائي لتمثيل ميل السطح الهندسي محلياً، مع إمكانية استقراء هذه القيم جزئياً من سحابة النقاط الهشة لتسريع التقارب.   
    
- **حساب التكلفة والتحويل المتجانس (Homography & Cost Computation):** يتم رسم نافذة مربعة (Patch) حول البكسل، وإسقاطها عبر تحويل متجانس يعتمد على العمق والاتجاه العمودي إلى الصورة المستهدفة، وتُحسب تكلفة التوافق البصري بالاعتماد على معامل الارتباط المتبادل الموحد (Normalized Cross Correlation - NCC).   
    
- **النشر الفضائي والتخصيص العشوائي (Spatial Propagation & Random Assignment):** تُنشر مستويات الأسطح ذات تكلفة NCC المنخفضة من البكسلات المجاورة المتصلة إلى البكسل الحالي لضمان نعومة السطح. تلي ذلك إضافة قيم عشوائية صغيرة لزوايا الميل والعمق لتفادي الوقوع في الحلول الصغرى المحلية.   
    
- **التصفية والدمج:** تُمحى نقاط العمق غير المستقرة فوتومترياً، وتُدمج خرائط العمق المتبقية بعد تصفيتها لبناء سحابة النقاط الكثيفة المصدرة في ملف `scene_dense.ply`.   
    

### 2. إعادة بناء السطح الشبكي (ReconstructMesh)

تقوم هذه الأداة باستقبال سحابة النقاط الكثيفة لتقدير سطح مثلثي مستمر ومغلق يغطي المجسم. تعتمد الخوارزمية بشكل أساسي على تقسيم الفضاء ثلاثي الأبعاد باستخدام شبكة رباعيات السطوح لديلاوني (Delaunay Tetrahedralization) لجميع نقاط السحابة الكثيفة. يتم صياغة مسألة تحديد السطح الحقيقي كمسألة تقسيم رسم بياني للتدفق الأقصى والقطع الأدنى (Min-Cut/Max-Flow Graph Cut).   

تُمثل خلايا رباعيات السطوح كعُقد في الرسم البياني، وتمثل الأوجه المشتركة بينها كحواف. يُصنف الفضاء الحجمي إلى خلايا تقع "داخل" المجسم المصمت (Inside) وخلايا تقع "خارج" المجسم في الفضاء الفارغ (Outside). تُحسب أوزان الحواف استناداً إلى تقاطع أشعة رؤية الكاميرات مع الأوجه الهندسية، حيث يمثل القطع الأدنى للرسم البياني القشرة السطحية الخارجية الأكثر اتساقاً هندسياً وفوتومترياً مع مسارات الرؤية.   

### 3. تحسين ودقة تفاصيل الشبكة (RefineMesh)

نظراً لكون السطح الشبكي المستخلص من خطوة القطع الأدنى للرسم البياني يحتوي على بعض التسطيح أو التشويه، تقوم أداة `RefineMesh` بإجراء تحسين دقيق للغاية على مواقع رؤوس المثلثات. يتم صياغة هذا التحسين كمسألة تقليل طاقة تعتمد على مشتقات الاتساق الفوتومتري للرقع السطحية للمثلثات عبر جميع الصور الملتقطة ذات الصلة، مما يعيد استخراج النتوءات والتفاصيل الدقيقة والأسطح المنحنية المعقدة بدقة بالغة.   

### 4. إكساء السطح بالنسيج الملون (TextureMesh)

للحصول على مجسم واقعي، تقوم أداة `TextureMesh` بإنشاء أطلس نسيجي ملون للشبكة الهندسية. تبدأ العملية بتحديد الكاميرا المثالية لإكساء كل مثلث في الشبكة. يتم صياغة هذه الخطوة كمسألة تعيين تسميات متعددة (Multi-Labeling) باستخدام حقل ماركوف العشوائي (Markov Random Field - MRF)، حيث يتم البحث عن التسمية الكلية التي تقلل التباين اللوني بين المثلثات المتجاورة المشتركة لمنع التقطع البصري، ويتم حل هذا النظام بكفاءة باستخدام خوارزمية تمرير الاعتقاد الحلقي (Loopy Belief Propagation - LBP).   

تلي ذلك خطوة معالجة الفواصل اللونية عبر تسوية الفوارق عالمياً (Global Seam Leveling) لتعديل فروق التعريض والإنارة الكلية بين الصور، ومحلياً (Local Seam Leveling) لدمج التدرجات اللونية عند خطوط التقاء رقع الإكساء المختلفة. ينتهي الأنبوب بتوليد ملف نسيجي موحد وإحداثيات إسقاط ثنائية الأبعاد (UV Coordinates) مرتبطة بالرؤوس الهندسية.   

## التكامل البرمجي والتحديات الهندسية المتقدمة

يكشف التطبيق العملي لدمج مكتبتي OpenMVG و OpenMVS في بيئات الإنتاج الفعلية عن مجموعة من العقبات التقنية والتحديات البرمجية التي تطلب صياغة حلول هندسية مبتكرة:

### معضلة الدقة العددية للقيم العائمة وإحداثيات الإسقاط الجغرافي

تعتبر مشكلة الدقة الحسابية للمتغيرات العائمة أحادية الدقة (`float`) واحدة من أكبر التحديات التشغيلية في مكتبة OpenMVS. ففي تطبيقات التصوير الجوي أو مسح خطوط الإنتاج الكبيرة باستخدام نقاط تحكم أرضية (GCPs) أو بيانات تحديد المواقع الجغرافية (GPS)، يتم تداول إحداثيات جغرافية حقيقية ضخمة تقع في نطاق ملايين الأمتار (مثل نظام إسقاط UTM).   

نظراً لكون البنية البرمجية لـ OpenMVS تعتمد على المتغيرات أحادية الدقة (`float`) لتسريع المعالجة الرسومية والرياضية، فإن هذه الأرقام الضخمة تؤدي إلى تلاشي الفروق العشرية الدقيقة للنقاط الكثيفة نتيجة الفقدان الكارثي للدقة الحسابية. يؤدي هذا التلاشي العددي هندسياً إلى تقليص سحابة النقاط الكثيفة وتشويهها لتظهر كخط مستقيم أحادي البعد، أو فشل خطوة بناء شبكة المثلثات `ReconstructMesh` تماماً وانهيار البرنامج صامتاً نتيجة قيم غير معرفة رياضياً (NAN) في مصفوفات تصفية ديلاوني.   

يظهر هذا التشوه أيضاً عند دمج بيانات GPS عشوائية في OpenMVG دون معالجة، مما ينتج مساحات شاسعة مشوهة باللون الأصفر أو خالية من النسيج اللوني في المجسم النهائي لـ OpenMVS. تقتضي المنهجية الهندسية الصارمة لمعالجة هذا الخلل القيام بتطبيع الإحداثيات (Normalization) بالكامل عند بداية خط أنابيب SfM في OpenMVG بحيث يتم إزاحة مركز المشهد ليتطابق مع النقطة الافتراضية (0,0,0) وضبط مقياسه الهندسي ليكون صغيراً. وتُحفظ مصفوفة الإزاحة والتحويل التشابهي (Similarity Transformation) بشكل مستقل، ليعاد تطبيقها وإسقاط الإحداثيات الجغرافية الحقيقية على المجسم بعد الانتهاء التام من جميع معالجات وعمليات OpenMVS.   

### عيوب إكساء النسيج الملون والأقنعة ومشاكل الانهيار البرمجي

عند التعامل مع مجموعات الصور التي تمتلك خلفيات معزولة باستخدام أقنعة ثنائية، يواجه المستخدمون مشكلة ظهور بقع هندسية سوداء مشوهة متناثرة على أسطح المجسمات عند تشغيل أداة `TextureMesh`. تنشأ هذه المشكلة نتيجة حدوث إزاحات طفيفة بين السطح الشبكي النهائي وحدود الأقنعة الثنائية المطبقة، مما يدفع خوارزمية الإكساء إلى سحب وتطبيق البكسلات السوداء للخلفية المعزولة على حواف المثلثات الهندسية.   

يكشف الفحص الدقيق لشفرة المصدر البرمجية لملف `SceneTexture.cpp` (تحديداً حول السطر 507) عن وجود خلل بنيوي عند محاولة تعطيل استخدام الأقنعة تماماً لتفادي البقع السوداء؛ حيث يؤدي تمرير المعامل `--ignore-mask-label` بقيمة أقل من -1 (مثل القيمة -2 لتعطيل الأقنعة بشكل كامل) إلى انهيار الأداة فجأة. يرجع هذا الانهيار البرمجي إلى عدم تهيئة متغير واجهة الإسقاط البصري للرقع (`rasterer.mask`)، مما يمنع تابع الإسقاط المباشر من إكمال حساباته التشغيلية.   

يتمثل الحل الهندسي لهذه المعضلة في إيقاف خيارات تسوية الفواصل المحلية والعالمية (`--local-seam-leveling 0` و `--global-seam-leveling 0`)؛ حيث يؤدي هذا الإيقاف إلى التخلص التام من تسريب اللون الأسود والبقع المشوهة والحصول على إكساء صحيح هندسياً، على الرغم من انخفاض الجودة اللونية الإجمالية عند حواف الالتقاء نتيجة غياب دمج التدرجات.   

### معوقات إدارة وتوسيع الذاكرة العشوائية الحجمية

تتسم عمليات مكتبة OpenMVS بحاجتها الشديدة لاهتمام خاص بإدارة الذاكرة العشوائية (RAM)، وتحديداً في خطوة `ReconstructMesh`. إن بناء شبكة رباعيات السطوح لديلاوني لعدد نقاط كثيف يتجاوز عشرات الملايين يستدعي بناء مصفوفة علاقات هندسية ضخمة الحجم تؤدي بالتبعية إلى نفاد الذاكرة العشوائية للجهاز وانهيار المعالجة في الأنظمة محدودة الموارد.   

لمعالجة هذا التقييد، تتيح المكتبة إمكانية خفض دقة الصور المرجعية أثناء التكثيف باستخدام معامل مستوى الدقة `--resolution-level` (مثل القيمة 2 أو 3) لتقليل الكثافة الإجمالية للنقاط. كما يجب الحذر عند إجراء أي تعديلات يدوية على سحابة النقاط الكثيفة المصدرة بصيغة `PLY` قبل تمريرها لأداة إعادة بناء الشبكة السطحية؛ حيث تطلب خوارزميات القطع الحجمي بقاء واحتفاظ جميع النقاط بخصائصها الفنية المحددة مثل اتجاهات أشعة الكاميرات المرتبطة وزوايا الرؤية، ويؤدي استخدام محرري النقاط الخارجيين الذين يهملون هذه البيانات الوصفية إلى فشل فوري ومباشر لعملية بناء السطح الشبكي.   

تتفوق البرمجيات التجارية الكبيرة مثل Agisoft Photoscan أو Bentley ContextCapture في قدرتها الحجمية لتداول هذه المشاهد الضخمة وتوسيع نطاق المعالجة الحسابية بفضل استخدام محركات معالجة حجمية مجزأة تلقائياً. وتوفر الجهة المطورة لمكتبة OpenMVS نظاماً تجارياً موازياً يسمى `3Dnovator` يقوم بأتمتة معالجة وتعديل الصور وإضاءتها، وإجراء تجزئة وتقسيم حجمي كلي للمشهد لتفادي قيود الذاكرة العشوائية والحصول على دقة موازية للتطبيقات الاحترافية في مجالات التصوير الجوي والمسح الجغرافي الواسع.   

### تقنيات الدمج المتقدمة وتداول النماذج الجاهزة (CAD models)

توفر مكتبة OpenMVS ميزة هندسية قوية غير موثقة بشكل كافٍ تتيح للمهندسين إكساء نماذج التصميم الهندسية الجاهزة (CAD Models) مباشرة بالاعتماد على وضعيات الكاميرات المقدرة رياضياً دون الحاجة لإعادة بناء المجسم من الصفر. تُعد هذه التقنية مثالية في قطاعات الفحص الهندسي الرقمي ومطابقة جودة المنتجات على خطوط الإنتاج بمقارنة النموذج الحقيقي بالـ CAD.   

لتنفيذ هذا الدمج، يتعين أولاً محاذاة نموذج الـ CAD مع نظام الإحداثيات الخاص بـ OpenMVG/OpenMVS باستخدام مصفوفة تحويل تشابهي تعتمد على مطابقة نقاط هندسية مميزة ثلاثية الأبعاد (يمكن حسابها وتطبيقها باستخدام برمجيات مثل MeshLab). بعد إتمام عملية التطابق والمحاذاة، يتم تفعيل الخيار المخفي للتحميل المباشر للشبكات في أداة `TextureMesh` عبر تمرير المعامل `--mesh-file` مقترناً بملف الـ CAD الخارجي (بتنسيق `OBJ` أو `PLY`) وملف المشهد المتكامل `scene_dense.mvs` الناتج عن الأنبوب الحسابي، ليقوم البرنامج بتوليد أطلس نسيجي عالي الدقة وتطبيقه على النموذج الهندسي الجاهز مباشرة دون المساس ببنيته الرياضية الأصلية.   

## مقارنة تقنية واستقصاء الأداء للأنظمة والخوارزميات

لتسهيل الفهم والمقارنة المباشرة بين الحلول البرمجية والهندسية المتوفرة في هاتين المكتبتين، تم تنظيم مقارنات تقنية مهيكلة بناءً على التحليل الحسابي لخصائص التشغيل والأداء الحقيقي للمنظومات الحسابية.

### أولاً: مقارنة خوارزميات حل البنية من الحركة (SfM) في OpenMVG

يوضح الجدول التالي الفروق الجوهرية والتشغيلية بين خطي الأنابيب التراكمي والعالمي المتاحين للباحثين والمهندسين داخل مكتبة OpenMVG:

|المعيار التقني والتشغيلي|خط الأنابيب التراكمي (Incremental SfM)|خط الأنابيب العالمي (Global SfM)|
|---|---|---|
|**آلية تسجيل وتوجيه الكاميرات**|تسجيل متتابع وتدريجي يبدأ بزوج صور مرجعي قوي، ثم تُضاف الكاميرات بالتناوب.|تقدير وحل متزامن وموحد لوضعيات جميع الكاميرات ونقاط البنية في خطوة تحسين مشتركة.|
|**الاستجابة لخطأ الانحراف البصري (Drift)**|عرضة بشكل كبير لتراكم أخطاء التقريب البصري والانحراف الهيدروليكي في المشاهد الطويلة والدائرية.|توزيع متساوٍ ومتجانس لجميع الأخطاء المتبقية، مما يمنع تراكم الانحراف الهندسي بفعالية.|
|**كفاءة المعالجة الزمنية والسرعة**|منخفضة السرعة؛ يتطلب تكرار عمليات التثليث وضبط حزم الأشعة الكلي مع كل كاميرا مضافة.|فائقة السرعة والتوفير في الموارد؛ حيث يتم تشغيل أداة ضبط الحزم لمرة واحدة في نهاية الأنبوب.|
|**المرونة مع البيانات المفقودة والمشوشة**|مرونة عالية جداً؛ القدرة على فحص الصور والتحقق من صحتها تدريجياً وبناء المشهد خطوة بخطوة.|حساسة ومحددة بمتانة مخطط الرؤية؛ تتطلب حسابات دوران وتراجم نسبية دقيقة وخالية من الشوائب.|
|**المكونات والحلول الرياضية الأساسية**|التثليث المحلي، تقدير الوضعية الفردية (Resection)، والضبط التكراري للمربعات الصغرى.|تصفية الاتساق الحلقي، حلول القيمة الذاتية للدورانات، ممتد ثلاثي البؤر، وتحسين L∞​ للتراجم.|

  

### ثانياً: الخصائص الفنية والتشغيلية لأدوات مكتبة OpenMVS الكثيفة

يوضح الجدول التالي تحليلاً تفصيلياً للأدوات التنفيذية الأربع المكونة لخط أنابيب OpenMVS، مبيناً خصائصها التشغيلية والبارامترية الأكثر أهمية:

|الأداة التنفيذية والبرمجية|مدخلات الأداة الأساسية|الخوارزمية الرياضية المطبقة|البارامترات الحاكمة والأخطاء الشائعة|مخرجات الأداة والملفات الناتجة|
|---|---|---|---|---|
|**DensifyPointCloud**<br><br>[cite: 25, 30]|ملف الكاميرات والوضعيات `scene.mvs` وسحابة النقاط الهشة.|خوارزمية مطابقة الرقع التصويرية العشوائية (PatchMatch Stereo) لخرائط العمق.|`--resolution-level` لخفض الجودة لتوفير الذاكرة. تنهار عند وجود إحداثيات جغرافية ضخمة.|ملف سحابة نقاط كثيفة دقيقة للغاية بتنسيق `scene_dense.ply`.|
|**ReconstructMesh**<br><br>[cite: 9, 25]|ملف سحابة النقاط الكثيفة المعدل أو الخام وصور الكاميرات.|تصفية خلايا ديلاوني ثلاثية الأبعاد (Delaunay Tetrahedralization) وحل القطع الأدنى للرسم البياني.|استهلاك فائق للذاكرة العشوائية (RAM). تفشل وتنهار صامتاً إذا حذفت الخصائص الهندسية للـ PLY.|ملف قشرة هندسية ثلاثية الأبعاد غير مكسوة بتنسيق `scene_dense_mesh.ply`.|
|**RefineMesh**<br><br>[cite: 5, 44]|ملف القشرة الهندسية المبدئية وسحابة النقاط الكثيفة والصور الأصلية.|تحسين موضعي غير خطي تكراري للرؤوس لتعظيم الاتساق الفوتومتري للأسطح.|تتطلب معالجة برمجية دقيقة للحدود والحواف لمنع التشوه الفغري غير الواقعي للأسطح.|شبكة مثلثات سطحية مطورة فائقة الدقة والبروزات الهندسية الدقيقة.|
|**TextureMesh**<br><br>[cite: 25, 35]|شبكة المثلثات النهائية المحسنة وصور الكاميرات والأقنعة الثنائية.|حل حقل ماركوف العشوائي بـ LBP لاختيار المشهد، المقترن بتسوية الفواصل الكلية والمحلية.|`--local-seam-leveling` يسبب بقعاً سوداء على الأسطح المقنعة. الخيار المخفي `--mesh-file` لإكساء CAD.|مجسم ثلاثي الأبعاد متكامل الإكساء بتنسيق `scene_dense_mesh_texture.ply/obj`.|

  

## استنتاجات وتوصيات هندسية للتطبيق الفعلي

يقدم الفحص التحليلي المتعمق للبنية الرياضية والبرمجية لمكتبتي OpenMVG و OpenMVS مجموعة من التوصيات الهندسية الحاسمة لضمان كفاءة معالجة البيانات ثلاثية الأبعاد وتجنب حالات الفشل البرمجي الشائعة:

- **أهمية عزل إحداثيات المشهد محلياً وتأجيل المعايرة الجغرافية:** يتضح علمياً أن المحافظة على معالجة سحب النقاط الكثيفة والشبكات الهندسية في نطاق إحداثيات محلي ممركز حول الصفر الافتراضي هو خط الدفاع الأول لمنع انهيارات الدقة العددية للأعداد العائمة أحادية الدقة في OpenMVS. يجب على المطورين الحفاظ على تقييد البيانات المصدرة من OpenMVG لتقع في نطاق هندسي صغير، والقيام بعمليات الإسقاط الجغرافي النهائي فقط كإجراء تجميلي لاحق بعد اكتمال المجسمات.   
    
- **هندسة تشغيل الأقنعة لعمليات الإكساء النظيف:** للتغلب على مشاكل البقع السوداء والتداخل اللوني السيئ في حواف المجسمات المعزولة، يُنصح بتعطيل معالجة تسوية الفواصل المحلية والعالمية أثناء استدعاء `TextureMesh` عند التعامل مع صور مقنعة غير متطابقة مئة بالمئة مع شبكة المثلثات الهندسية. كما يجب تجنب تمرير قيم لتعطيل الأقنعة تقل عن -1 لعدم وقوع انهيارات الذاكرة البرمجية غير المعرفة في الكود الداخلي للمكتبة.   
    
- **التوسيع والاستعانة بالحلول الموازية للمشاهد الضخمة:** نظراً لكون الذاكرة العشوائية هي العامل المحدد للقدرة الحجمية في OpenMVS (خصوصاً لتقسيم خلايا ديلاوني ورسومات القطع الموصولة للشبكات)، يتعين على المهندسين استكشاف تجزئة المشاهد الحجمية الكبيرة مسبقاً إلى كتل مستقلة تتم معالجتها وحسابها بشكل منفرد ثم إعادة ربطها، أو النظر في تبني الحلول البرمجية القادرة على الإدارة الديناميكية للذاكرة للمشاهد الممتدة.   
    
- **الآفاق المستقبلية واستيعاب عدم اليقين (Anisotropic Models):** يتجه التطوير البحثي المعاصر للرؤية الحاسوبية نحو تبني أطر تقدير الدوران غير المتجانس (Anisotropic Rotation Averaging)، حيث لا تكتفي الخوارزميات بتقدير زوايا الدوران المطلقة للكاميرات فحسب، بل يتم دمج مصفوفات عدم اليقين (Uncertainty Matrices) الناتجة عن المطابقات ثنائية الصور مباشرة داخل أطر تحسين المصفوفات شبه المعرفة (SDP Relaxations) لإنتاج حلول شديدة المتانة والدقة للمشاهد المعقدة والمشوشة. وتفتح هذه التطورات الرياضية الباب واسعاً لبناء جيل جديد من برمجيات إعادة البناء ثلاثي الأبعاد قادرة على مجاراة أعلى مستويات الدقة المطلوبة عالمياً في قطاعات الهندسة والصناعة.   
    

[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2407.20219v2)

arxiv.org

Global Structure-from-Motion Revisited - arXiv

Opens in a new window](https://arxiv.org/html/2407.20219v2)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fisprs-archives.copernicus.org%2Farticles%2FXLII-2-W15%2F1135%2F2019%2Fisprs-archives-XLII-2-W15-1135-2019.pdf)

isprs-archives.copernicus.org

MULTI-VIEW STEREO WITH SEMANTIC PRIORS - ISPRS - The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences

Opens in a new window](https://isprs-archives.copernicus.org/articles/XLII-2-W15/1135/2019/isprs-archives-XLII-2-W15-1135-2019.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2604.10246v1)

arxiv.org

A Comparison of Multi-View Stereo Methods for Photogrammetric 3D Reconstruction: From Traditional to Learning-Based Approaches - arXiv

Opens in a new window](https://arxiv.org/html/2604.10246v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fopenmvg%2Fopenmvg)

github.com

OpenMVG (open Multiple View Geometry) - GitHub

Opens in a new window](https://github.com/openmvg/openmvg)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fcdcseacave.github.io%2F)

cdcseacave.github.io

OpenMVS by cdcseacave

Opens in a new window](https://cdcseacave.github.io/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenmvg.readthedocs.io%2F)

openmvg.readthedocs.io

openMVG documentation! — openMVG library

Opens in a new window](https://openmvg.readthedocs.io/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fscholar.google.com%2Fcitations%3Fuser%3D8u1nmVUAAAAJ%26hl%3Dfr)

scholar.google.com

‪Pierre Moulon‬ - ‪Google Scholar‬

Opens in a new window](https://scholar.google.com/citations?user=8u1nmVUAAAAJ&hl=fr)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fhost139.ik.w-hs.de%2Ftsky%2Fvcpkg%2F-%2Fblob%2Fe2049cb9754006b6a2abed781d34030e16702fad%2Fports%2Fopenmvs%2Fportfile.cmake)

host139.ik.w-hs.de

ports/openmvs/portfile.cmake · e2049cb9754006b6a2abed781d34030e16702fad · Tom _sky / vcpkg - GitLab

Opens in a new window](https://host139.ik.w-hs.de/tsky/vcpkg/-/blob/e2049cb9754006b6a2abed781d34030e16702fad/ports/openmvs/portfile.cmake)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F1158)

github.com

How to perform “ReconstructMesh” after editing the “scene_dense.ply” file · Issue #1158 · cdcseacave/openMVS - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/1158)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FopenMVG%2FopenMVG%2Freleases)

github.com

Releases · openMVG/openMVG - GitHub

Opens in a new window](https://github.com/openMVG/openMVG/releases)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenmvg.readthedocs.io%2Fen%2Flatest%2Fsoftware%2FSfM%2FSfM%2F)

openmvg.readthedocs.io

SfM: Structure from Motion — openMVG library - Read the Docs

Opens in a new window](https://openmvg.readthedocs.io/en/latest/software/SfM/SfM/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.sfpt.fr%2Fwp-content%2Fuploads%2F2016%2F03%2F20160315_Tutoriel_OpenMVG_SFPT-2016.pdf)

sfpt.fr

OpenMVG Tutorial

Opens in a new window](https://www.sfpt.fr/wp-content/uploads/2016/03/20160315_Tutoriel_OpenMVG_SFPT-2016.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2F%40lienghongky%2Fsfm-3d-reconstruction-with-openmvg-and-openmvs-1ad81c3743d0)

medium.com

SFM: 3D Reconstruction with openMVG and openMVS | by Z-9 - Medium

Opens in a new window](https://medium.com/@lienghongky/sfm-3d-reconstruction-with-openmvg-and-openmvs-1ad81c3743d0)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenmvg.readthedocs.io%2Fen%2Flatest%2Fsoftware%2FSfM%2FComputeFeatures%2F)

openmvg.readthedocs.io

openMVG_main_ComputeFeatur

Opens in a new window](https://openmvg.readthedocs.io/en/latest/software/SfM/ComputeFeatures/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenmvg.readthedocs.io%2Fen%2Flatest%2FnonFree%2Fpatented%2F)

openmvg.readthedocs.io

patented — openMVG library - Read the Docs

Opens in a new window](https://openmvg.readthedocs.io/en/latest/nonFree/patented/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FopenMVG%2FopenMVG%2Fissues%2F654)

github.com

Understanding OpenMVG features · Issue #654 - GitHub

Opens in a new window](https://github.com/openMVG/openMVG/issues/654)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fdiscussions%2F1173)

github.com

PointCloud from drone aerial images · cdcseacave openMVS · Discussion #1173 - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/discussions/1173)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC8465351%2F)

pmc.ncbi.nlm.nih.gov

Comparative Analysis of Detectors and Feature Descriptors for Multispectral Image Matching in Rice Crops - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC8465351/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FopenMVG%2FopenMVG%2Fissues%2F1361)

github.com

openMVG2openMVS toward the inside out · Issue #1361 - GitHub

Opens in a new window](https://github.com/openMVG/openMVG/issues/1361)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F232747391_Automatic_Homographic_Registration_of_a_Pair_of_Images_with_A_Contrario_Elimination_of_Outliers)

researchgate.net

Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/232747391_Automatic_Homographic_Registration_of_a_Pair_of_Images_with_A_Contrario_Elimination_of_Outliers)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fimagine.enpc.fr%2F%7Emoulonp%2Fpublis%2FACCV12_Demo_Poster.pdf)

imagine.enpc.fr

Adaptive model estimation, a real time demonstration - Imagine

Opens in a new window](http://imagine.enpc.fr/~moulonp/publis/ACCV12_Demo_Poster.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fimagine.enpc.fr%2F%7Emoulonp%2Fpublis%2Ficcv2013%2Findex.html)

imagine.enpc.fr

Global Fusion of Relative Motions for Robust, Accurate and Scalable Structure from Motion

Opens in a new window](http://imagine.enpc.fr/~moulonp/publis/iccv2013/index.html)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdemuc.de%2Fpapers%2Fpan2024glomap.pdf)

demuc.de

Global Structure-from-Motion Revisited - Johannes Schönberger

Opens in a new window](https://demuc.de/papers/pan2024glomap.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fceres-solver.org%2Fusers.html)

ceres-solver.org

Users - Ceres Solver

Opens in a new window](http://ceres-solver.org/users.html)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenmvg.readthedocs.io%2Fen%2Flatest%2Fsoftware%2FMVS%2FOpenMVS%2F)

openmvg.readthedocs.io

OpenMVS Open Multiple View Stereovision — openMVG library - Read the Docs

Opens in a new window](https://openmvg.readthedocs.io/en/latest/software/MVS/OpenMVS/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F179)

github.com

Using openMVS as a third party library · Issue #179 - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/179)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F356)

github.com

ReconstructMesh exits after delaunay tetrahedras weighting · Issue #356 · cdcseacave/openMVS - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/356)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgitee.com%2FLiuxinqi12%2FopenMVS%2Fblob%2Fmaster%2FBUILD.md)

gitee.com

Liuxinqi12/openMVS - Gitee

Opens in a new window](https://gitee.com/Liuxinqi12/openMVS/blob/master/BUILD.md)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fh005.github.io%2F2017%2F09%2F04%2FopenMVG-and-openMVS%2F)

h005.github.io

3D Reconstruction with openMVG and openMVS - - hejw005 blog

Opens in a new window](https://h005.github.io/2017/09/04/openMVG-and-openMVS/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Faashutosh-py.medium.com%2Funderstanding-the-dense-point-cloud-generation-densifypointcloud-in-openmvs-faa6777c7501)

aashutosh-py.medium.com

Understanding the Dense Point-Cloud Generation (DensifyPointCloud) in OpenMVS

Opens in a new window](https://aashutosh-py.medium.com/understanding-the-dense-point-cloud-generation-densifypointcloud-in-openmvs-faa6777c7501)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F382)

github.com

How DensifyPointCloud does dense reconstruction from sparse reconstruction? · Issue #382 · cdcseacave/openMVS - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/382)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2301.05206)

arxiv.org

arXiv:2301.05206v1 [cs.RO] 12 Jan 2023

Opens in a new window](https://arxiv.org/pdf/2301.05206)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.scribd.com%2Fdocument%2F615821541%2F%25E8%25A7%2586%25E8%25A7%2589%25E5%2587%25A0%25E4%25BD%2595%25E4%25B8%2589%25E7%25BB%25B4%25E9%2587%258D%25E5%25BB%25BA-OpenMVS%25E6%25BA%2590%25E7%25A0%2581%25E8%25A7%25A3%25E6%259E%2590)

scribd.com

视觉几何三维重建OpenMVS源码解析| PDF - Scribd

Opens in a new window](https://www.scribd.com/document/615821541/%E8%A7%86%E8%A7%89%E5%87%A0%E4%BD%95%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA-OpenMVS%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Far5iv.labs.arxiv.org%2Fhtml%2F2301.05206)

ar5iv.labs.arxiv.org

[2301.05206] ImMesh: An Immediate LiDAR Localization and Meshing Framework - ar5iv

Opens in a new window](https://ar5iv.labs.arxiv.org/html/2301.05206)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F1251)

github.com

[OpenMVS] TextureMesh creates black artifacts/patches on the object surface (masked dataset) · Issue #1251 - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/1251)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F99)

github.com

problem with DensifyPointCloud and ReconstructMesh · Issue #99 · cdcseacave/openMVS

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/99)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F263)

github.com

[Question] Poor quality of Mesh with ReconstructMesh command. How to improve it? · Issue #263 · cdcseacave/openMVS - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/263)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F318)

github.com

Some texture is missing after running TextureMesh · Issue #318 · cdcseacave/openMVS

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/318)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2FopenMVG%2FopenMVG%2Fissues%2F1405)

github.com

Advice regarding bundle adjustment based on covariance matrices · Issue #1405 - GitHub

Opens in a new window](https://github.com/openMVG/openMVG/issues/1405)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgroups.google.com%2Fg%2Fopenmvs%2Fc%2F_8wwpqWPnQk)

groups.google.com

Quality, memory and speed settings in OpenMVS - Google Groups

Opens in a new window](https://groups.google.com/g/openmvs/c/_8wwpqWPnQk)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F243)

github.com

Texture a different mesh using OpenMVG+OpenMVS pipeline · Issue #243 - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/243)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fninercommons.charlotte.edu%2Frecord%2F2173%2Ffiles%2FNawab_uncc_0694D_13269.pdf)

ninercommons.charlotte.edu

AUGMENTED REALITY WITH DIGITAL METROLOGY FOR ASSEMBLY by Rahma Nawab - Niner Commons - UNC Charlotte

Opens in a new window](https://ninercommons.charlotte.edu/record/2173/files/Nawab_uncc_0694D_13269.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F337)

github.com

I want to run ReconstructMesh, how should I get scene_dense.mvs?(Assume I only have sfm_data.bin) · Issue #337 · cdcseacave/openMVS - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/337)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fgithub.com%2Fcdcseacave%2FopenMVS%2Fissues%2F1142)

github.com

Output of ReconstructMesh · Issue #1142 · cdcseacave/openMVS - GitHub

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/1142)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2503.07353)

arxiv.org

arXiv:2503.07353v1 [cs.CV] 10 Mar 2025

Opens in a new window](https://arxiv.org/pdf/2503.07353)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FICCV2025%2Fpapers%2FOlsson_Certifiably_Optimal_Anisotropic_Rotation_Averaging_ICCV_2025_paper.pdf)

openaccess.thecvf.com

Certifiably Optimal Anisotropic Rotation Averaging - CVF Open Access

Opens in a new window](https://openaccess.thecvf.com/content/ICCV2025/papers/Olsson_Certifiably_Optimal_Anisotropic_Rotation_Averaging_ICCV_2025_paper.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2506.01940)

arxiv.org

Making Rotation Averaging Fast and Robust with Anisotropic Coordinate Descent - arXiv

Opens in a new window](https://arxiv.org/pdf/2506.01940)

[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/openMVG/openMVG/wiki)[

![](https://t1.gstatic.com/faviconV2?url=https://openmvg.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openmvg.readthedocs.io/en/latest/software/MVS/MVS/)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/openMVG/openMVG/issues/2192)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/openMVG/openMVG/issues/910)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/openMVG/openMVG/issues/2059)[

![](https://t0.gstatic.com/faviconV2?url=https://groups.google.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://groups.google.com/g/openmvs/c/RLRsK-GLrzw)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/534)[

![](https://t3.gstatic.com/faviconV2?url=https://pdfs.semanticscholar.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://pdfs.semanticscholar.org/f8d3/5e5c04933e6cb76e027f31c04ecc3f9eb92c.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://www.ipol.im/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.ipol.im/pub/art/2019/248/article.pdf)[

![](https://t0.gstatic.com/faviconV2?url=https://scholar.google.co.id/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://scholar.google.co.id/citations?user=dZRK3HsAAAAJ&hl=tr)[

![](https://t1.gstatic.com/faviconV2?url=https://scholar.google.fi/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://scholar.google.fi/citations?user=8u1nmVUAAAAJ&hl=zh-TW)[

![](https://t1.gstatic.com/faviconV2?url=https://openmvg.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openmvg.readthedocs.io/en/latest/bibliography/)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/972)[

![](https://t2.gstatic.com/faviconV2?url=http://liu.diva-portal.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](http://liu.diva-portal.org/smash/record.jsf?pid=diva2:345932)[

![](https://t2.gstatic.com/faviconV2?url=https://isy.gitlab-pages.liu.se/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://isy.gitlab-pages.liu.se/cvl/en/courses/TSBB33/project.html)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/ceres-solver/ceres-solver/issues/428)[

![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://stackoverflow.com/questions/48041209/ceres-solver-bundle-adjustment)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/105)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/231)[

![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://github.com/cdcseacave/openMVS/issues/1140)[

![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.mdpi.com/2072-4292/14/15/3521)[

![](https://t2.gstatic.com/faviconV2?url=https://scispace.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://scispace.com/pdf/collaborative-localization-for-micro-aerial-vehicles-1y9vc7y7ic.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://dspace.lib.ntua.gr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dspace.lib.ntua.gr/xmlui/bitstream/handle/123456789/56640/PhD_Stathopoulou_Integrating_scene_priors_in_MVS.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openaccess.thecvf.com/content/CVPR2024/papers/Tirado-Garin_From_Correspondences_to_Pose_Non-minimal_Certifiably_Optimal_Relative_Pose_without_CVPR_2024_paper.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://research.chalmers.se/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://research.chalmers.se/publication/549568/file/549568_Fulltext.pdf)
