"?f
BHostIDLE"IDLE1    ?|?@A    ?|?@a䒣? ;??i䒣? ;???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a????m.??i??R?`???Unknown?
iHostWriteSummary"WriteSummary(1     ?D@9     ?D@A     ?D@I     ?D@a???|?m?i?H?f?~???Unknown?
uHost_FusedMatMul"sequential_18/dense_60/Relu(1      B@9      B@A      B@I      B@a??j	?j?i>??F?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?@@9     ?@@A     ?@@I     ?@@a?+?]??g?ij??Ӝ????Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1     ?@@9     ?@@A     ?@@I     ?@@a?+?]??g?i??Va?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?A@9     ?A@A      >@I      >@aO???:?e?i2%	??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      6@9      6@A      6@I      6@a?:?'g?_?iO?O/????Unknown
x	HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?Q@9     ?Q@A      1@I      1@a?[?A??X?i}??????Unknown
?
HostMatMul"-gradient_tape/sequential_18/dense_61/MatMul_1(1      &@9      &@A      &@I      &@a?:?'g?O?i? ??w???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a{????L?ik,n?	???Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@a??j	?J?i?p	;???Unknown
dHostDataset"Iterator::Model(1      E@9      E@A      "@I      "@a??j	?J?i??r????Unknown
HostMatMul"+gradient_tape/sequential_18/dense_61/MatMul(1      "@9      "@A      "@I      "@a??j	?J?ix<u?D???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      0@9      0@A       @I       @a?%z.G?i?œ?#???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a?%z.G?ivO??(???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?%z.G?i??Ў?.???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?%z.G?itb?s4???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a?%z.G?i???>:???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?%z.G?iru,$
@???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a?%z.G?i??J??E???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_62/MatMul(1       @9       @A       @I       @a?%z.G?ip?i2?K???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?%z.G?i???lQ???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?<??XHD?i>???~V???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?<??XHD?i?????[???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_60/MatMul(1      @9      @A      @I      @a?<??XHD?i?:8??`???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a}?[?bA?i?!???d???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a}?[?bA?i	?FTi???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a}?[?bA?i9?<??m???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a}?[?bA?iXד?r???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_62/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a}?[?bA?iw??6^v???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a{????<?if?]k?y???Unknown
?!HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a{????<?iU?П?}???Unknown
?"HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a{????<?iD D?;????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a{????<?i3?ۄ???Unknown
v$HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a{????<?i",*=z????Unknown
?%HostBiasAddGrad"8gradient_tape/sequential_18/dense_61/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a{????<?iB?q????Unknown
x&Host_FusedMatMul"sequential_18/dense_62/BiasAdd(1      @9      @A      @I      @a{????<?i X??????Unknown
s'HostSigmoid"sequential_18/dense_62/Sigmoid(1      @9      @A      @I      @a{????<?i?m??W????Unknown
X(HostCast"Cast_5(1      @9      @A      @I      @a?%z.7?i???=????Unknown
V)HostMean"Mean(1      @9      @A      @I      @a?%z.7?io??a#????Unknown
u*HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?%z.7?i/<1%	????Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?%z.7?i?????????Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?%z.7?i??O?ԡ???Unknown
?-HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?%z.7?io
?o?????Unknown
?.HostBiasAddGrad"8gradient_tape/sequential_18/dense_60/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?%z.7?i/On3?????Unknown
?/HostMatMul"-gradient_tape/sequential_18/dense_62/MatMul_1(1      @9      @A      @I      @a?%z.7?i?????????Unknown
u0Host_FusedMatMul"sequential_18/dense_61/Relu(1      @9      @A      @I      @a?%z.7?i?،?k????Unknown
?1HostReadVariableOp",sequential_18/dense_62/MatMul/ReadVariableOp(1      @9      @A      @I      @a?%z.7?io~Q????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a}?[?b1?i????}????Unknown
X3HostCast"Cast_3(1      @9      @A      @I      @a}?[?b1?i?s#?????Unknown
X4HostEqual"Equal(1      @9      @A      @I      @a}?[?b1?ixvֶ???Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a}?[?b1?i????????Unknown
?6HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      @I      @a}?[?b1?i?_u/????Unknown
s7HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a}?[?b1?i?? n[????Unknown
|8HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a}?[?b1?i_F???????Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a}?[?b1?i??w?????Unknown
~:HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a}?[?b1?i-#f?????Unknown
b;HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a}?[?b1?i?θ????Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a}?[?b1?i?z9????Unknown
~=HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a}?[?b1?i/?%^e????Unknown
?>HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a}?[?b1?i??а?????Unknown
??Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a}?[?b1?iOo|?????Unknown
?@Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a}?[?b1?i??'V?????Unknown
?AHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a}?[?b1?ioVӨ????Unknown
?BHostReadVariableOp"-sequential_18/dense_60/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a}?[?b1?i??~?B????Unknown
?CHostReadVariableOp"-sequential_18/dense_61/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a}?[?b1?i?=*No????Unknown
tDHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?%z.'?i???/?????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?%z.'?iO??U????Unknown
VFHostCast"Cast(1       @9       @A       @I       @a?%z.'?i?$???????Unknown
TGHostMul"Mul(1       @9       @A       @I       @a?%z.'?i?H?:????Unknown
dHHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?%z.'?ioi??????Unknown
jIHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?%z.'?i?ؘ ????Unknown
rJHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?%z.'?i/??z?????Unknown
vKHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?%z.'?i?Pg\????Unknown
?LHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?%z.'?i??.>y????Unknown
vMHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?%z.'?iO???????Unknown
?NHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?%z.'?i?7?_????Unknown
`OHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?%z.'?iڅ??????Unknown
xPHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?%z.'?io|M?D????Unknown
?QHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?%z.'?i???????Unknown
?RHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?%z.'?i/?܈*????Unknown
?SHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?%z.'?i?c?j?????Unknown
?THostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?%z.'?i?lL????Unknown
~UHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?%z.'?iO?3.?????Unknown
?VHostReluGrad"-gradient_tape/sequential_18/dense_60/ReluGrad(1       @9       @A       @I       @a?%z.'?i?J??????Unknown
?WHostReluGrad"-gradient_tape/sequential_18/dense_61/ReluGrad(1       @9       @A       @I       @a?%z.'?i???h????Unknown
?XHostReadVariableOp",sequential_18/dense_60/MatMul/ReadVariableOp(1       @9       @A       @I       @a?%z.'?io????????Unknown
?YHostReadVariableOp",sequential_18/dense_61/MatMul/ReadVariableOp(1       @9       @A       @I       @a?%z.'?i?1R?N????Unknown
vZHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?%z.?i?6&????Unknown
X[HostCast"Cast_4(1      ??9      ??A      ??I      ??a?%z.?i/???????Unknown
a\HostIdentity"Identity(1      ??9      ??A      ??I      ??a?%z.?i_??{????Unknown?
}]HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?%z.?i?v?x4????Unknown
u^HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?%z.?i?G???????Unknown
w_HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?%z.?i??Z?????Unknown
?`HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?%z.?i???`????Unknown
?aHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?%z.?iO?p<????Unknown
?bHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?%z.?i?T??????Unknown
?cHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?%z.?i?]8?????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?%z.?i?.?F????Unknown
?eHostReadVariableOp"-sequential_18/dense_62/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?%z.?i     ???Unknown
LfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
WgHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i     ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
YiHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown*?f
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a??W??;??i??W??;???Unknown?
iHostWriteSummary"WriteSummary(1     ?D@9     ?D@A     ?D@I     ?D@aP?1;????i?31?8???Unknown?
uHost_FusedMatMul"sequential_18/dense_60/Relu(1      B@9      B@A      B@I      B@aк?????iXA׈????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?@@9     ?@@A     ?@@I     ?@@a镱??^??i?? {????Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1     ?@@9     ?@@A     ?@@I     ?@@a镱??^??i?Z?xo????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?A@9     ?A@A      >@I      >@aq??$??i>f???c???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      6@9      6@A      6@I      6@a?!?????i?n0E>????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?Q@9     ?Q@A      1@I      1@a6?a]#??i<?E??S???Unknown
?	HostMatMul"-gradient_tape/sequential_18/dense_61/MatMul_1(1      &@9      &@A      &@I      &@a?!?????ivy??r????Unknown
?
HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a??31?~?iN}Q??????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@aк???{?i? {?L???Unknown
dHostDataset"Iterator::Model(1      E@9      E@A      "@I      "@aк???{?i:??ҦC???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_61/MatMul(1      "@9      "@A      "@I      "@aк???{?i??? {???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      0@9      0@A       @I       @a?????x?i?
?F4????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a?????x?i???g????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?????x?i??J????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?????x?i?~??????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a?????x?ijNq???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?????x?i"V?5????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a?????x?i5BRi????Unknown
HostMatMul"+gradient_tape/sequential_18/dense_62/MatMul(1       @9       @A       @I       @a?????x?iH .Ԝ???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?????x?i[#V?5???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aiXA׈?u?i??g?`???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aiXA׈?u?i?(wy?????Unknown
HostMatMul"+gradient_tape/sequential_18/dense_60/MatMul(1      @9      @A      @I      @aiXA׈?u?in?%??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a5'??Psr?i???,?????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a5'??Psr?i
??? ???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a5'??Psr?iX?xo?%???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a5'??Psr?i????J???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_62/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a5'??Psr?i??Z?xo???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??31?n?i?8??8????Unknown
? HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a??31?n?i̺??????Unknown
?!HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??31?n?i?<?E?????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??31?n?i??(wy????Unknown
v#HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??31?n?i?@\?9	???Unknown
?$HostBiasAddGrad"8gradient_tape/sequential_18/dense_61/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??31?n?i|??'???Unknown
x%Host_FusedMatMul"sequential_18/dense_62/BiasAdd(1      @9      @A      @I      @a??31?n?ihD?
?F???Unknown
s&HostSigmoid"sequential_18/dense_62/Sigmoid(1      @9      @A      @I      @a??31?n?iT??;ze???Unknown
X'HostCast"Cast_5(1      @9      @A      @I      @a?????h?i????~???Unknown
V(HostMean"Mean(1      @9      @A      @I      @a?????h?ih?⽭????Unknown
u)HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?????h?i???~G????Unknown
v*HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?????h?i|????????Unknown
?+HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?????h?i?? {????Unknown
?,HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?????h?i?Ϻ?????Unknown
?-HostBiasAddGrad"8gradient_tape/sequential_18/dense_60/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?????h?iѰ?????Unknown
?.HostMatMul"-gradient_tape/sequential_18/dense_62/MatMul_1(1      @9      @A      @I      @a?????h?i?ҦCH*???Unknown
u/Host_FusedMatMul"sequential_18/dense_61/Relu(1      @9      @A      @I      @a?????h?i.Ԝ?B???Unknown
?0HostReadVariableOp",sequential_18/dense_62/MatMul/ReadVariableOp(1      @9      @A      @I      @a?????h?i?Ւ?{[???Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a5'??Psb?i?VK?m???Unknown
X2HostCast"Cast_3(1      @9      @A      @I      @a5'??Psb?i?gb????Unknown
X3HostEqual"Equal(1      @9      @A      @I      @a5'??Psb?i-Y??Ւ???Unknown
\4HostGreater"Greater(1      @9      @A      @I      @a5'??Psb?iT?tI????Unknown
?5HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      @I      @a5'??Psb?i{[-Y?????Unknown
s6HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a5'??Psb?i????/????Unknown
|7HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a5'??Psb?i?]???????Unknown
z8HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a5'??Psb?i??VK????Unknown
~9HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a5'??Psb?i`?????Unknown
b:HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a5'??Psb?i>???????Unknown
w;HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a5'??Psb?ieb?=p&???Unknown
~<HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a5'??Psb?i??8??8???Unknown
?=HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a5'??Psb?i?d??VK???Unknown
?>Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a5'??Psb?i???/?]???Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a5'??Psb?igb?=p???Unknown
?@HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a5'??Psb?i(?Ѱ????Unknown
?AHostReadVariableOp"-sequential_18/dense_60/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a5'??Psb?iOi?!$????Unknown
?BHostReadVariableOp"-sequential_18/dense_61/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a5'??Psb?iv??r?????Unknown
tCHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?????X?i;?S?????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?????X?i ??31????Unknown
VEHostCast"Cast(1       @9       @A       @I       @a?????X?i???~????Unknown
TFHostMul"Mul(1       @9       @A       @I       @a?????X?i??w??????Unknown
dGHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?????X?iO???????Unknown
jHHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?????X?i?m?d????Unknown
rIHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?????X?i??蕱????Unknown
vJHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?????X?i??cv?	???Unknown
?KHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?????X?ic??VK???Unknown
vLHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?????X?i(?Y7?"???Unknown
?MHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?????X?i????.???Unknown
`NHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?????X?i??O?1;???Unknown
xOHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?????X?iw???~G???Unknown
?PHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?????X?i<?E??S???Unknown
?QHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?????X?i???`???Unknown
?RHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?????X?i??;zel???Unknown
?SHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?????X?i???Z?x???Unknown
~THostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?????X?iP?1;?????Unknown
?UHostReluGrad"-gradient_tape/sequential_18/dense_60/ReluGrad(1       @9       @A       @I       @a?????X?i??L????Unknown
?VHostReluGrad"-gradient_tape/sequential_18/dense_61/ReluGrad(1       @9       @A       @I       @a?????X?i??'??????Unknown
?WHostReadVariableOp",sequential_18/dense_60/MatMul/ReadVariableOp(1       @9       @A       @I       @a?????X?i?????????Unknown
?XHostReadVariableOp",sequential_18/dense_61/MatMul/ReadVariableOp(1       @9       @A       @I       @a?????X?id??2????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?????H?i?{[-Y????Unknown
XZHostCast"Cast_4(1      ??9      ??A      ??I      ??a?????H?i(???????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a?????H?i?|??????Unknown?
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?????H?i??~?????Unknown
u]HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?????H?iN}Q??????Unknown
w^HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?????H?i???^????Unknown
?_HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?????H?i~???????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?????H?it?	?f????Unknown
?aHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?????H?i?~G??????Unknown
?bHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?????H?i8???????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?????H?i??????Unknown
?dHostReadVariableOp"-sequential_18/dense_62/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?????H?i?????????Unknown
LeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
WfHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU