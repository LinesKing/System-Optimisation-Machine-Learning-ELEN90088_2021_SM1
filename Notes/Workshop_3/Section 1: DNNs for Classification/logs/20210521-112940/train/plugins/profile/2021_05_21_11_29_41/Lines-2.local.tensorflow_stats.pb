"?d
BHostIDLE"IDLE1     ??@A     ??@a???| ,??i???| ,???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      W@9      W@A      W@I      W@a?(?ˈ??i?J??f????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?T@9     ?T@A     ?T@I     ?T@ai}??????i?68;6????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1     ?N@9     ?N@A     ?N@I     ?N@a?* cf5??ij??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      B@9      B@A      B@I      B@a??Bk??i?J??V???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?@@9     ?@@A     ?@@I     ?@@a/k?g"???i_<p?@????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?B@9     ?B@A      @@I      @@a?H)_??i?M?;?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      <@9      <@A      <@I      <@a?G???|?i???K
???Unknown
i	HostWriteSummary"WriteSummary(1      4@9      4@A      4@I      4@a"[?f?vt?i8g?2?=???Unknown?
d
HostDataset"Iterator::Model(1      [@9      [@A      0@I      0@a?H)_p?i?o䄶^???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      .@9      .@A      .@I      .@a?(m?n?iӗ??h}???Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@a?䥊؂f?i?=???????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a??Bkb?i?"??V????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a??Bkb?i[??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a??Bkb?i-?Q?,????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      "@9      "@A      "@I      "@a??Bkb?i?Д?????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?H)_`?iHճ,?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a?H)_`?i???UV????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?H)_`?i???~????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?G???\?i??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?G???\?iZ%?\+???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?G???\?iI?J?9???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a?G???\?i?lގH???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a\m????X?i???IT???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a\m????X?iH??L?`???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a\m????X?i6d??l???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a\m????X?i?y;
 y???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a\m????X?i??ig????Unknown
VHostCast"Cast(1      @9      @A      @I      @a"[?f?vT?i??⢏???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a"[?f?vT?iI?y\ޙ???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a"[?f?vT?i??,?????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a"[?f?vT?i?G?OU????Unknown
?!HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a"[?f?vT?iS??ɐ????Unknown
?"HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a"[?f?vT?iGC?????Unknown
?#HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a"[?f?vT?i?o??????Unknown
?$HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a"[?f?vT?i]ҭ6C????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?H)_P?i?T=?r????Unknown
\&HostGreater"Greater(1      @9      @A      @I      @a?H)_P?i???_?????Unknown
e'Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?H)_P?i?X\??????Unknown?
?(HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?H)_P?i????????Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a?H)_P?i]{1 ???Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?H)_P?i5?
?`???Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?H)_P?iYa?F????Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?H)_P?i}?)ۿ???Unknown
v-HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?H)_P?i?e?o? ???Unknown
?.HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?H)_P?i??H)???Unknown
?/HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?H)_P?i?iؘN1???Unknown
}0HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?H)_P?i?g-~9???Unknown
X1HostCast"Cast_3(1      @9      @A      @I      @a\m????H?i???ܡ????Unknown
V2HostMean"Mean(1      @9      @A      @I      @a\m????H?iC/???E???Unknown
?3HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a\m????H?i?Ъ;?K???Unknown
?4HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a\m????H?iyr?R???Unknown
v5HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a\m????H?i??0X???Unknown
b6HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a\m????H?i???IT^???Unknown
?7HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a\m????H?iJWY?wd???Unknown
?8Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a\m????H?i??Ĩ?j???Unknown
?9HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a\m????H?i??0X?p???Unknown
?:HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a\m????H?i<??v???Unknown
?;HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a\m????H?i???}???Unknown
o<HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a\m????H?iQsf*????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?H)_@?ic@?0B????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?H)_@?iu?Y????Unknown
X?HostCast"Cast_4(1       @9       @A       @I       @a?H)_@?i??J?q????Unknown
X@HostCast"Cast_5(1       @9       @A       @I       @a?H)_@?i?????????Unknown
|AHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?H)_@?i?D?Y?????Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?H)_@?i?"$?????Unknown
rCHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?H)_@?i??i?П???Unknown
vDHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?H)_@?iᇱ??????Unknown
?EHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?H)_@?i?H?? ????Unknown
`FHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?H)_@?i
AM????Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?H)_@?iˈ0????Unknown
~HHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?H)_@?i)???G????Unknown
?IHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?H)_@?i;M?_????Unknown
?JHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?H)_@?iM`vw????Unknown
?KHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?H)_@?i_ϧ@?????Unknown
?LHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?H)_@?iq??
?????Unknown
?MHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?H)_@?i?Q7վ????Unknown
~NHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?H)_@?i???????Unknown
?OHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?H)_@?i???i?????Unknown
}PHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?H)_@?i??4????Unknown
QHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?H)_@?i?UV?????Unknown
?RHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a?H)_@?i???5????Unknown
?SHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?H)_@?i????M????Unknown
tTHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a?H)_0?ix?	xY????Unknown
TUHostMul"Mul(1      ??9      ??A      ??I      ??a?H)_0?i?-]e????Unknown
sVHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a?H)_0?i?yQBq????Unknown
uWHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?H)_0?iZu'}????Unknown
jXHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a?H)_0?i?:??????Unknown
}YHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?H)_0?i%???????Unknown
wZHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?H)_0?i???֠????Unknown
?[HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?H)_0?i7???????Unknown
x\HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?H)_0?i??(??????Unknown
?]HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?H)_0?iI?L??????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?H)_0?i?}pk?????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?H)_0?i[^?P?????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?H)_0?i?>?5?????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?H)_0?im??????Unknown
?bHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?H)_0?i?????????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?H)_0?i@??????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?H)_0?i??#????Unknown
IeHostAssignAddVariableOp"AssignAddVariableOp_3(i??#????Unknown
4fHostIdentity"Identity(i??#????Unknown?
JgHostReadVariableOp"div_no_nan/ReadVariableOp_1(i??#????Unknown
LhHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i??#????Unknown*?c
sHostDataset"Iterator::Model::ParallelMapV2(1      W@9      W@A      W@I      W@a??3??i??3???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?T@9     ?T@A     ?T@I     ?T@aj?큷??i??_*?w???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1     ?N@9     ?N@A     ?N@I     ?N@a9'???\??i4????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      B@9      B@A      B@I      B@a?`?r?6??iI?l~?????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?@@9     ?@@A     ?@@I     ?@@a?X?(??i\????????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?B@9     ?B@A      @@I      @@a@?u?i??i?p??j???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      <@9      <@A      <@I      <@a??f?????iOg[a????Unknown
iHostWriteSummary"WriteSummary(1      4@9      4@A      4@I      4@a??3??ip@XLR????Unknown?
d	HostDataset"Iterator::Model(1      [@9      [@A      0@I      0@a@?u?i??i???d?????Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      .@9      .@A      .@I      .@a?Pn?&??iN`כ?*???Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@axˡ6lю?i|??L?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?`?r?6??i??{:?
???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?`?r?6??i?
F(?o???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a?`?r?6??i?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      "@9      "@A      "@I      "@a?`?r?6??i?-?o9???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a@?u?i??i5????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a@?u?i??i??]?????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a@?u?i??i???(dF???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??f?????ifMYS֔???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??f?????i=?~H????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??f?????i?̨?1???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a??f?????i? ??,????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @ap@XLRπ?i큷j????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ap@XLRπ?i???e????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @ap@XLRπ?i?C??I???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ap@XLRπ?i??K?!????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @ap@XLRπ?i?}A_????Unknown
VHostCast"Cast(1      @9      @A      @I      @a??3|?i!,&?g???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a??3|?iMR?p@???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??3|?iyxxxxx???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??3|?i??!??????Unknown
? HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??3|?i???G?????Unknown
?!HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??3|?i??s?? ???Unknown
?"HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??3|?i)?X???Unknown
?#HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a??3|?iU7?~?????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a@?u?iv?i?"?v????Unknown
\%HostGreater"Greater(1      @9      @A      @I      @a@?u?iv?i?I????Unknown
e&Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a@?u?iv?iX?(???Unknown?
?'HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a@?u?iv?i??I??C???Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a@?u?iv?i?j?p???Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a@?u?iv?iZ????????Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a@?u?iv?i???)k????Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a@?u?iv?i?ͯ>????Unknown
v,HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a@?u?iv?i\}?5$???Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a@?u?iv?i?h??P???Unknown
?.HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a@?u?iv?iT0B?}???Unknown
}/HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a@?u?iv?i^?QȌ????Unknown
X0HostCast"Cast_3(1      @9      @A      @I      @ap@XLR?p?i???l+????Unknown
V1HostMean"Mean(1      @9      @A      @I      @ap@XLR?p?i`???????Unknown
?2HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ap@XLR?p?i?P?h???Unknown
?3HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @ap@XLR?p?ib?Z1???Unknown
v4HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @ap@XLR?p?i??L??R???Unknown
b5HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @ap@XLR?p?idb??Dt???Unknown
?6HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @ap@XLR?p?i?~H?????Unknown
?7Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @ap@XLR?p?if?큷???Unknown
?8HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @ap@XLR?p?i?s?? ????Unknown
?9HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @ap@XLR?p?ih$H6?????Unknown
?:HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @ap@XLR?p?i????]???Unknown
o;HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @ap@XLR?p?ij?y?=???Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a@?u?if?i??BfT???Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a@?u?if?i?p??j???Unknown
X>HostCast"Cast_4(1       @9       @A       @I       @a@?u?if?ik???9????Unknown
X?HostCast"Cast_5(1       @9       @A       @I       @a@?u?if?i\???????Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a@?u?if?i???N????Unknown
dAHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a@?u?if?ilG?w????Unknown
rBHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a@?u?if?i????????Unknown
vCHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a@?u?if?i?2??J????Unknown
?DHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a@?u?if?im?[????Unknown
`EHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a@?u?if?i???Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a@?u?if?iÓ.??4???Unknown
~GHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a@?u?if?in	???J???Unknown
?HHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a@?u?if?iOg[a???Unknown
?IHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a@?u?if?i??_*?w???Unknown
?JHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a@?u?if?iojp?.????Unknown
?KHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a@?u?if?i????????Unknown
?LHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a@?u?if?i?U?s????Unknown
~MHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a@?u?if?ipˡ6l????Unknown
?NHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a@?u?if?iA???????Unknown
}OHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a@?u?if?iƶ¼?????Unknown
PHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a@?u?if?iq,?????Unknown
?QHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a@?u?if?i??B+???Unknown
?RHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a@?u?if?i??}A???Unknown
tSHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a@?u?iV?i?R|??L???Unknown
TTHostMul"Mul(1      ??9      ??A      ??I      ??a@?u?iV?is???W???Unknown
sUHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a@?u?iV?iIȌ?c???Unknown
uVHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a@?u?iV?i?Pn???Unknown
jWHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a@?u?iV?i?=?m?y???Unknown
}XHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a@?u?iV?i?x%O?????Unknown
wYHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a@?u?iV?i???0?????Unknown
?ZHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a@?u?iV?iw?5$????Unknown
x[HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a@?u?iV?iM)??X????Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a@?u?iV?i#dFՍ????Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a@?u?iV?i??ζ¼???Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a@?u?iV?i??V??????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a@?u?iV?i??y,????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a@?u?iV?i{Og[a????Unknown
?aHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a@?u?iV?iQ??<?????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a@?u?iV?i'?w?????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a@?u?iV?i?????????Unknown
IdHostAssignAddVariableOp"AssignAddVariableOp_3(i?????????Unknown
4eHostIdentity"Identity(i?????????Unknown?
JfHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
LgHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown2CPU