"?n
BHostIDLE"IDLE1     J?@A     J?@ay? -p???iy? -p????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a?y ?,???i????i???Unknown?
iHostWriteSummary"WriteSummary(1      C@9      C@A      C@I      C@a?Ʃ??l?is?n?Ӆ???Unknown?
uHost_FusedMatMul"sequential_12/dense_39/Relu(1      B@9      B@A      B@I      B@a??P?:k?i??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?D@9     ?D@A      ;@I      ;@a?`???kd?iU???y????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      3@I      3@a?Ʃ??\?i?u???????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      2@9      2@A      2@I      2@a??P?:[?i???v????Unknown
gHostStridedSlice"strided_slice(1      1@9      1@A      1@I      1@a?@ڲ??Y?ie?Q????Unknown
`	HostGatherV2"
GatherV2_1(1      ,@9      ,@A      ,@I      ,@a??w?-U?i?FEc?????Unknown
?
HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ,@9      @A      ,@I      @a??w?-U?i??%#????Unknown
VHostCast"Cast(1      *@9      *@A      *@I      *@a?@???S?i6CT????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      (@9      (@A      (@I      (@a? ???&R?i????g???Unknown
HostMatMul"+gradient_tape/sequential_12/dense_40/MatMul(1      $@9      $@A      $@I      $@a=??@N?i??????Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a??P?:K?i???????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??P?:K?i6@?B????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a? d??3H?i6?B?!???Unknown
dHostDataset"Iterator::Model(1      :@9      :@A       @I       @a? d??3H?i6??B?'???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a? d??3H?i6?vB?-???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a? d??3H?i6?dB?3???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a? d??3H?i6}RB?9???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a? d??3H?i6V@B?????Unknown
?HostMatMul"-gradient_tape/sequential_12/dense_40/MatMul_1(1       @9       @A       @I       @a? d??3H?i6/.B?E???Unknown
HostMatMul"+gradient_tape/sequential_12/dense_41/MatMul(1       @9       @A       @I       @a? d??3H?i6B?K???Unknown
?HostMatMul"-gradient_tape/sequential_12/dense_41/MatMul_1(1       @9       @A       @I       @a? d??3H?i6?	B
R???Unknown
HostMatMul"+gradient_tape/sequential_12/dense_42/MatMul(1       @9       @A       @I       @a? d??3H?i6??AX???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??w?-E?i???b]???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a? ???&B?i?:?a?a???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a? ???&B?i???!vf???Unknown
HostMatMul"+gradient_tape/sequential_12/dense_39/MatMul(1      @9      @A      @I      @a? ???&B?iV ???j???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_12/dense_42/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a? ???&B?ic???o???Unknown
uHost_FusedMatMul"sequential_12/dense_40/Relu(1      @9      @A      @I      @a? ???&B?i?ţat???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a=??@>?ivm???w???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a=??@>?i???{???Unknown
v"HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a=??@>?i????k???Unknown
~#HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a=??@>?iVdv?3????Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a=??@>?i?k?????Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a=??@>?i??_!Ċ???Unknown
?&HostBiasAddGrad"8gradient_tape/sequential_12/dense_40/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a=??@>?i6[TA?????Unknown
?'HostBiasAddGrad"8gradient_tape/sequential_12/dense_41/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a=??@>?i?IaT????Unknown
s(HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a? d??38?iV???Z????Unknown
?)HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a? d??38?i??6aa????Unknown
?*HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a? d??38?iV?-?g????Unknown
V+HostSum"Sum_2(1      @9      @A      @I      @a? d??38?iִ$an????Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a? d??38?iV??t????Unknown
?-Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a? d??38?i֍a{????Unknown
?.HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a? d??38?iVz	ၧ???Unknown
u/Host_FusedMatMul"sequential_12/dense_41/Relu(1      @9      @A      @I      @a? d??38?i?f a?????Unknown
x0Host_FusedMatMul"sequential_12/dense_42/BiasAdd(1      @9      @A      @I      @a? d??38?iVS???????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a? ???&2?i????ӯ???Unknown
V2HostMean"Mean(1      @9      @A      @I      @a? ???&2?i???????Unknown
?3HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a? ???&2?iv???]????Unknown
?4HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a? ???&2?i??`?????Unknown
z5HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a? ???&2?i6J?@?????Unknown
v6HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a? ???&2?i?{? ,????Unknown
b7HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a? ???&2?i??? q????Unknown
?8HostBiasAddGrad"8gradient_tape/sequential_12/dense_39/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a? ???&2?iV??ൿ???Unknown
?9HostMatMul"-gradient_tape/sequential_12/dense_42/MatMul_1(1      @9      @A      @I      @a? ???&2?i????????Unknown
?:HostReadVariableOp"-sequential_12/dense_40/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a? ???&2?iA???????Unknown
s;HostSigmoid"sequential_12/dense_42/Sigmoid(1      @9      @A      @I      @a? ???&2?ivr???????Unknown
t<HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a? d??3(?i????????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a? d??3(?i?^? ?????Unknown
X>HostCast"Cast_3(1       @9       @A       @I       @a? d??3(?i6՞@????Unknown
X?HostEqual"Equal(1       @9       @A       @I       @a? d??3(?ivK???????Unknown
\@HostGreater"Greater(1       @9       @A       @I       @a? d??3(?i????????Unknown
uAHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a? d??3(?i?7? ?????Unknown
|BHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a? d??3(?i6??@????Unknown
dCHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a? d??3(?iv$???????Unknown
jDHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a? d??3(?i????!????Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a? d??3(?i? ?????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a? d??3(?i6?z@(????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a? d??3(?iv?u??????Unknown
`HHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a? d??3(?i?sq?.????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a? d??3(?i??l ?????Unknown
xJHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a? d??3(?i6`h@5????Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a? d??3(?iv?c??????Unknown
?LHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a? d??3(?i?L_?;????Unknown
?MHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a? d??3(?i??Z ?????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a? d??3(?i69V@B????Unknown
?OHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a? d??3(?iv?Q??????Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a? d??3(?i?%M?H????Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a? d??3(?i??H ?????Unknown
?RHostReluGrad"-gradient_tape/sequential_12/dense_40/ReluGrad(1       @9       @A       @I       @a? d??3(?i6D@O????Unknown
?SHostReadVariableOp"-sequential_12/dense_39/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a? d??3(?iv????????Unknown
?THostReadVariableOp",sequential_12/dense_39/MatMul/ReadVariableOp(1       @9       @A       @I       @a? d??3(?i??:?U????Unknown
?UHostReadVariableOp"-sequential_12/dense_41/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a? d??3(?i?t6 ?????Unknown
?VHostReadVariableOp"-sequential_12/dense_42/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a? d??3(?i6?1@\????Unknown
?WHostReadVariableOp",sequential_12/dense_42/MatMul/ReadVariableOp(1       @9       @A       @I       @a? d??3(?iva-??????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a? d??3?i?+ ?????Unknown
XYHostCast"Cast_4(1      ??9      ??A      ??I      ??a? d??3?i??(?b????Unknown
XZHostCast"Cast_5(1      ??9      ??A      ??I      ??a? d??3?i֒&`$????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a? d??3?i?M$ ?????Unknown?
v\HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a? d??3?i	"??????Unknown
}]HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a? d??3?i6?@i????Unknown
w^HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a? d??3?iV?*????Unknown
w_HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a? d??3?iv:??????Unknown
y`HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a? d??3?i?? ?????Unknown
?aHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a? d??3?i???o????Unknown
?bHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a? d??3?i?k`1????Unknown
?cHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a? d??3?i?& ?????Unknown
?dHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a? d??3?i???????Unknown
?eHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a? d??3?i6?@v????Unknown
?fHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a? d??3?iVX?7????Unknown
?gHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a? d??3?iv	??????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a? d??3?i?? ?????Unknown
?iHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a? d??3?i???|????Unknown
?jHostReluGrad"-gradient_tape/sequential_12/dense_39/ReluGrad(1      ??9      ??A      ??I      ??a? d??3?i?D`>????Unknown
?kHostReluGrad"-gradient_tape/sequential_12/dense_41/ReluGrad(1      ??9      ??A      ??I      ??a? d??3?i?????????Unknown
?lHostReadVariableOp",sequential_12/dense_40/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a? d??3?i????` ???Unknown
?mHostReadVariableOp",sequential_12/dense_41/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a? d??3?i???? ???Unknown
'nHostMul"Mul(i???? ???Unknown
WoHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i???? ???Unknown
YpHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i???? ???Unknown
[qHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i???? ???Unknown*?n
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a???4?<??i???4?<???Unknown?
iHostWriteSummary"WriteSummary(1      C@9      C@A      C@I      C@aa??}??i???3????Unknown?
uHost_FusedMatMul"sequential_12/dense_39/Relu(1      B@9      B@A      B@I      B@a???(9??i?26?O????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?D@9     ?D@A      ;@I      ;@aK???*Ғ?id?+T?o???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      3@I      3@aa??}??iJb???????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      2@9      2@A      2@I      2@a???(9??i12`?6>???Unknown
gHostStridedSlice"strided_slice(1      1@9      1@A      1@I      1@a??mR???iZ????Unknown
`HostGatherV2"
GatherV2_1(1      ,@9      ,@A      ,@I      ,@a?K<????i?{????Unknown
?	HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ,@9      @A      ,@I      @a?K<????i????(9???Unknown
V
HostCast"Cast(1      *@9      *@A      *@I      *@ax?!????i?A?ѧ????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      (@9      (@A      (@I      (@a????к??i?!?????Unknown
HostMatMul"+gradient_tape/sequential_12/dense_40/MatMul(1      $@9      $@A      $@I      $@a?G??{?i??T"W????Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a???(9y?i?????.???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a???(9y?i????`???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @al???kNv?i??]?T????Unknown
dHostDataset"Iterator::Model(1      :@9      :@A       @I       @al???kNv?i?õ?????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @al???kNv?i?A(??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @al???kNv?iz??d+???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @al???kNv?io??;?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @al???kNv?idXel???Unknown
?HostMatMul"-gradient_tape/sequential_12/dense_40/MatMul_1(1       @9       @A       @I       @al???kNv?iYA??????Unknown
HostMatMul"+gradient_tape/sequential_12/dense_41/MatMul(1       @9       @A       @I       @al???kNv?iN?"????Unknown
?HostMatMul"-gradient_tape/sequential_12/dense_41/MatMul_1(1       @9       @A       @I       @al???kNv?iC???;????Unknown
HostMatMul"+gradient_tape/sequential_12/dense_42/MatMul(1       @9       @A       @I       @al???kNv?i8?p????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?K<??s?i.?e??E???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a????кp?i&??NWg???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a????кp?iy}?̈???Unknown
HostMatMul"+gradient_tape/sequential_12/dense_39/MatMul(1      @9      @A      @I      @a????кp?ii	?B????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_12/dense_42/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????кp?iY?3?????Unknown
uHost_FusedMatMul"sequential_12/dense_40/Relu(1      @9      @A      @I      @a????кp?iI!?-????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?G??k?i????	???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a?G??k?i??_??$???Unknown
v!HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?G??k?i? ???@???Unknown
~"HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?G??k?i?h???\???Unknown
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?G??k?i??=??x???Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?G??k?i????y????Unknown
?%HostBiasAddGrad"8gradient_tape/sequential_12/dense_40/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?G??k?i?@|\????Unknown
?&HostBiasAddGrad"8gradient_tape/sequential_12/dense_41/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?G??k?iΈ
>????Unknown
s'HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @al???kNf?i?(?u?????Unknown
?(HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @al???kNf?i?Ȁ??????Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @al???kNf?i?h3M)???Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @al???kNf?i???w%???Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @al???kNf?i???$?;???Unknown
?,Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @al???kNf?i?HK?R???Unknown
?-HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @al???kNf?i????bh???Unknown
u.Host_FusedMatMul"sequential_12/dense_41/Relu(1      @9      @A      @I      @al???kNf?i???g?~???Unknown
x/Host_FusedMatMul"sequential_12/dense_42/BiasAdd(1      @9      @A      @I      @al???kNf?i?(c??????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a????к`?i? )??????Unknown
V1HostMean"Mean(1      @9      @A      @I      @a????к`?i??tu????Unknown
?2HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a????к`?i??E0????Unknown
?3HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a????к`?i?{?????Unknown
z4HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a????к`?i? A??????Unknown
v5HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a????к`?i???`????Unknown
b6HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a????к`?i|?̈
???Unknown
?7HostBiasAddGrad"8gradient_tape/sequential_12/dense_39/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????к`?ix??Y????Unknown
?8HostMatMul"-gradient_tape/sequential_12/dense_42/MatMul_1(1      @9      @A      @I      @a????к`?it?X*?+???Unknown
?9HostReadVariableOp"-sequential_12/dense_40/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????к`?ip??K<???Unknown
s:HostSigmoid"sequential_12/dense_42/Sigmoid(1      @9      @A      @I      @a????к`?il???M???Unknown
t;HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @al???kNV?ii ?.X???Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @al???kNV?ifp?7Uc???Unknown
X=HostCast"Cast_3(1       @9       @A       @I       @al???kNV?ic?pm|n???Unknown
X>HostEqual"Equal(1       @9       @A       @I       @al???kNV?i`J??y???Unknown
\?HostGreater"Greater(1       @9       @A       @I       @al???kNV?i]`#?ʄ???Unknown
u@HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @al???kNV?iZ???????Unknown
|AHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @al???kNV?iW ?D????Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @al???kNV?iTP?z@????Unknown
jCHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @al???kNV?iQ???g????Unknown
rDHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @al???kNV?iN?a掼???Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @al???kNV?iK@;?????Unknown
?FHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @al???kNV?iH?R?????Unknown
`GHostDivNoNan"
div_no_nan(1       @9       @A       @I       @al???kNV?iE???????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @al???kNV?iB0ǽ+????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @al???kNV?i????R????Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @al???kNV?i<?y)z????Unknown
?KHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @al???kNV?i9 S_?
???Unknown
?LHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @al???kNV?i6p,?????Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @al???kNV?i3??? ???Unknown
?NHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @al???kNV?i0? ,???Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @al???kNV?i-`?6>7???Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @al???kNV?i*??leB???Unknown
?QHostReluGrad"-gradient_tape/sequential_12/dense_40/ReluGrad(1       @9       @A       @I       @al???kNV?i' k??M???Unknown
?RHostReadVariableOp"-sequential_12/dense_39/BiasAdd/ReadVariableOp(1       @9       @A       @I       @al???kNV?i$PDسX???Unknown
?SHostReadVariableOp",sequential_12/dense_39/MatMul/ReadVariableOp(1       @9       @A       @I       @al???kNV?i!??c???Unknown
?THostReadVariableOp"-sequential_12/dense_41/BiasAdd/ReadVariableOp(1       @9       @A       @I       @al???kNV?i??Co???Unknown
?UHostReadVariableOp"-sequential_12/dense_42/BiasAdd/ReadVariableOp(1       @9       @A       @I       @al???kNV?i@?y)z???Unknown
?VHostReadVariableOp",sequential_12/dense_42/MatMul/ReadVariableOp(1       @9       @A       @I       @al???kNV?i???P????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??al???kNF?i8?J?????Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??al???kNF?i???w????Unknown
XYHostCast"Cast_5(1      ??9      ??A      ??I      ??al???kNF?i?o?????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??al???kNF?i0\?????Unknown?
v[HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??al???kNF?i?H?2????Unknown
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??al???kNF?i?5QƦ???Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??al???kNF?i("?Y????Unknown
w^HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??al???kNF?i???????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??al???kNF?ix?!?????Unknown
?`HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??al???kNF?i ??????Unknown
?aHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??al???kNF?i??W?????Unknown
?bHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??al???kNF?ip??;????Unknown
?cHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??al???kNF?i???????Unknown
?dHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??al???kNF?i
??(c????Unknown
?eHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??al???kNF?i	h???????Unknown
?fHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??al???kNF?it^?????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??al???kNF?i?`?????Unknown
?hHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??al???kNF?i`M??????Unknown
?iHostReluGrad"-gradient_tape/sequential_12/dense_39/ReluGrad(1      ??9      ??A      ??I      ??al???kNF?i:/E????Unknown
?jHostReluGrad"-gradient_tape/sequential_12/dense_41/ReluGrad(1      ??9      ??A      ??I      ??al???kNF?i?&??????Unknown
?kHostReadVariableOp",sequential_12/dense_40/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??al???kNF?iXel????Unknown
?lHostReadVariableOp",sequential_12/dense_41/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??al???kNF?i     ???Unknown
'mHostMul"Mul(i     ???Unknown
WnHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
YoHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[pHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown2CPU