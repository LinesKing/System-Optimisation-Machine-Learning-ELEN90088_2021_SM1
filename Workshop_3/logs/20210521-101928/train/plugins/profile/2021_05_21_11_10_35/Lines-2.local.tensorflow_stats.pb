"?Q
BHostIDLE"IDLE1     λ@A     λ@a5?0????i5?0?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?????i?5????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?B@9     ?B@A     ?B@I     ?B@a7???wr?i)??D???Unknown
lHostIteratorGetNext"IteratorGetNext(1     ?A@9     ?A@A     ?A@I     ?A@a?{	X[xq?i?0١?g???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ;@9      ;@A      :@I      :@ao?????i?i??F?????Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@ao?????i?i ??뛛???Unknown?
vHost_FusedMatMul"sequential_59/dense_183/Relu(1      7@9      7@A      7@I      7@aY?e??f?i??-??????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      4@I      4@aB?S?C?c?i?? $?????Unknown
^	HostGatherV2"GatherV2(1      1@9      1@A      1@I      1@a+/????`?i????????Unknown
`
HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a??}???]?i?*{????Unknown
vHost_FusedMatMul"sequential_59/dense_184/Relu(1      (@9      (@A      (@I      (@a?`????W?i=?h?u????Unknown
jHostMean"binary_crossentropy/Mean(1      &@9      &@A      &@I      &@a?-)4d?U?iԦ?q????Unknown
bHostDivNoNan"div_no_nan_1(1      &@9      &@A      &@I      &@a?-)4d?U?ik??Pl???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aB?S?C?S?ii???g???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a??~p#?Q?i?$>d???Unknown
dHostDataset"Iterator::Model(1     ?F@9     ?F@A       @I       @a?+S?O?i?yŅ`#???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?+S?O?ic?L]+???Unknown?
?HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?+S?O?i.#ԈY3???Unknown
?HostMatMul".gradient_tape/sequential_59/dense_184/MatMul_1(1       @9       @A       @I       @a?+S?O?i?w[
V;???Unknown
?HostMatMul",gradient_tape/sequential_59/dense_185/MatMul(1       @9       @A       @I       @a?+S?O?i????RC???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_59/dense_185/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)ƨY??K?i?69}OJ???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a?`????G?i??^?LP???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`????G?i&6??JV???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`????G?i????G\???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`????G?iV5?Eb???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`????G?i???bBh???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?`????G?i?4??n???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?`????G?i??%=t???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?`????G?i?3e?:z???Unknown
?HostMatMul",gradient_tape/sequential_59/dense_183/MatMul(1      @9      @A      @I      @a?`????G?iN???7????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aB?S?C?C?iMH?5????Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?+S???i??B?3????Unknown
V!HostMean"Mean(1      @9      @A      @I      @a?+S???i?:2????Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @a?+S???i|G?z0????Unknown
v#HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?+S???i????.????Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?+S???iF?Q?,????Unknown
?%HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?+S???i?F=+????Unknown
?&HostBiasAddGrad"9gradient_tape/sequential_59/dense_184/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?+S???i??})????Unknown
?'HostMatMul",gradient_tape/sequential_59/dense_184/MatMul(1      @9      @A      @I      @a?+S???iu???'????Unknown
?(HostReadVariableOp".sequential_59/dense_184/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?+S???i?E`?%????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?`????7?i???$????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?`????7?irŅ`#????Unknown
\+HostGreater"Greater(1      @9      @A      @I      @a?`????7?i>?"????Unknown
s,HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?`????7?i
E?? ????Unknown
u-HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?`????7?i?>r????Unknown
?.HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?`????7?i???"????Unknown
v/HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a?`????7?in?c?????Unknown
?0HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?`????7?i:D??????Unknown
v1HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?`????7?i?4????Unknown
v2HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?`????7?i???????Unknown
?3HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      @9      @A      @I      @a?`????7?i????????Unknown
?4HostBiasAddGrad"9gradient_tape/sequential_59/dense_183/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?`????7?ijCAF????Unknown
?5HostMatMul".gradient_tape/sequential_59/dense_185/MatMul_1(1      @9      @A      @I      @a?`????7?i6??????Unknown
?6HostReadVariableOp".sequential_59/dense_183/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?`????7?i?f?????Unknown
y7Host_FusedMatMul"sequential_59/dense_185/BiasAdd(1      @9      @A      @I      @a?`????7?i΂?W????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?+S?/?iX[x????Unknown
X9HostCast"Cast_3(1       @9       @A       @I       @a?+S?/?i4-??????Unknown
X:HostCast"Cast_4(1       @9       @A       @I       @a?+S?/?ig?????Unknown
X;HostEqual"Equal(1       @9       @A       @I       @a?+S?/?i?׀?????Unknown
d<HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?+S?/?iͬ??????Unknown
r=HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?+S?/?i ?D????Unknown
`>HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?+S?/?i3W?:????Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?+S?/?if,[????Unknown
?@Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?+S?/?i?j{
????Unknown
?AHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a?+S?/?i??˛	????Unknown
?BHostReluGrad".gradient_tape/sequential_59/dense_184/ReluGrad(1       @9       @A       @I       @a?+S?/?i??-?????Unknown
?CHostReadVariableOp".sequential_59/dense_185/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?+S?/?i2???????Unknown
?DHostReadVariableOp"-sequential_59/dense_185/MatMul/ReadVariableOp(1       @9       @A       @I       @a?+S?/?ieV??????Unknown
tEHostSigmoid"sequential_59/dense_185/Sigmoid(1       @9       @A       @I       @a?+S?/?i?+S????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?+S??i1??????Unknown
aGHostIdentity"Identity(1      ??9      ??A      ??I      ??a?+S??i? ?=????Unknown?
?HHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?+S??ic???????Unknown
TIHostMul"Mul(1      ??9      ??A      ??I      ??a?+S??i??^????Unknown
|JHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?+S??i??G?????Unknown
vKHostMul"%binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?+S??i.?x~????Unknown
}LHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?+S??iǕ?????Unknown
uMHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?+S??i`?ڞ????Unknown
wNHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?+S??i?j/????Unknown
yOHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?+S??i?U<?????Unknown
?PHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?+S??i+@mO????Unknown
?QHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?+S??i?*?? ????Unknown
?RHostReadVariableOp"-sequential_59/dense_183/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?+S??i]?o ????Unknown
?SHostReadVariableOp"-sequential_59/dense_184/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?+S??i?????????Unknown
WTHostReluGrad".gradient_tape/sequential_59/dense_183/ReluGrad(i?????????Unknown*?Q
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@aZZZZZ???iZZZZZ????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?B@9     ?B@A     ?B@I     ?B@aiiiiii??i?????????Unknown
lHostIteratorGetNext"IteratorGetNext(1     ?A@9     ?A@A     ?A@I     ?A@axxxxxx??ixxxxx????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ;@9      ;@A      :@I      :@axxxxxx??i<<<<<????Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@axxxxxx??i     ????Unknown?
vHost_FusedMatMul"sequential_59/dense_183/Relu(1      7@9      7@A      7@I      7@a????????i------???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      4@I      4@a?????Ғ?i?????????Unknown
^HostGatherV2"GatherV2(1      1@9      1@A      1@I      1@a      ??i?????C???Unknown
`	HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a<<<<<<??i?????????Unknown
v
Host_FusedMatMul"sequential_59/dense_184/Relu(1      (@9      (@A      (@I      (@a????????i???Unknown
jHostMean"binary_crossentropy/Mean(1      &@9      &@A      &@I      &@a????????i?????a???Unknown
bHostDivNoNan"div_no_nan_1(1      &@9      &@A      &@I      &@a????????i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?????҂?i      ???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a????????i?????C???Unknown
dHostDataset"Iterator::Model(1     ?F@9     ?F@A       @I       @a~?i     ????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a~?i<<<<<????Unknown?
?HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a~?ixxxxx????Unknown
?HostMatMul".gradient_tape/sequential_59/dense_184/MatMul_1(1       @9       @A       @I       @a~?i?????4???Unknown
?HostMatMul",gradient_tape/sequential_59/dense_185/MatMul(1       @9       @A       @I       @a~?i?????p???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_59/dense_185/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZZZZZZz?i?????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a??????v?i?????????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????v?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????v?i,-----???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????v?iYZZZZZ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????v?i?????????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??????v?i?????????Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??????v?i?????????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??????v?i???Unknown
?HostMatMul",gradient_tape/sequential_59/dense_183/MatMul(1      @9      @A      @I      @a??????v?i:<<<<<???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??????r?i?????a???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @an?i????????Unknown
V HostMean"Mean(1      @9      @A      @I      @an?i????Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @an?i:<<<<????Unknown
v"HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @an?iXZZZZ????Unknown
~#HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @an?ivxxxx????Unknown
?$HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @an?i????????Unknown
?%HostBiasAddGrad"9gradient_tape/sequential_59/dense_184/BiasAdd/BiasAddGrad(1      @9      @A      @I      @an?i?????4???Unknown
?&HostMatMul",gradient_tape/sequential_59/dense_184/MatMul(1      @9      @A      @I      @an?i?????R???Unknown
?'HostReadVariableOp".sequential_59/dense_184/BiasAdd/ReadVariableOp(1      @9      @A      @I      @an?i?????p???Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??????f?i?????????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??????f?i????Unknown
\*HostGreater"Greater(1      @9      @A      @I      @a??????f?i?????????Unknown
s+HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??????f?iJKKKK????Unknown
u,HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??????f?i?????????Unknown
?-HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??????f?ixxxxx????Unknown
v.HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a??????f?i???Unknown
?/HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??????f?i?????%???Unknown
v0HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??????f?i=<<<<<???Unknown
v1HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??????f?i?????R???Unknown
?2HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      @9      @A      @I      @a??????f?ikiiiii???Unknown
?3HostBiasAddGrad"9gradient_tape/sequential_59/dense_183/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??????f?i    ????Unknown
?4HostMatMul".gradient_tape/sequential_59/dense_185/MatMul_1(1      @9      @A      @I      @a??????f?i?????????Unknown
?5HostReadVariableOp".sequential_59/dense_183/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??????f?i0----????Unknown
y6Host_FusedMatMul"sequential_59/dense_185/BiasAdd(1      @9      @A      @I      @a??????f?i?????????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a^?i?????????Unknown
X8HostCast"Cast_3(1       @9       @A       @I       @a^?i?????????Unknown
X9HostCast"Cast_4(1       @9       @A       @I       @a^?i?????????Unknown
X:HostEqual"Equal(1       @9       @A       @I       @a^?i     ???Unknown
d;HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a^?i???Unknown
r<HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a^?i!???Unknown
`=HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a^?i0-----???Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a^?i?<<<<<???Unknown
??Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a^?iNKKKKK???Unknown
?@HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a^?i]ZZZZZ???Unknown
?AHostReluGrad".gradient_tape/sequential_59/dense_184/ReluGrad(1       @9       @A       @I       @a^?iliiiii???Unknown
?BHostReadVariableOp".sequential_59/dense_185/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a^?i{xxxxx???Unknown
?CHostReadVariableOp"-sequential_59/dense_185/MatMul/ReadVariableOp(1       @9       @A       @I       @a^?i?????????Unknown
tDHostSigmoid"sequential_59/dense_185/Sigmoid(1       @9       @A       @I       @a^?i?????????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aN?i!????Unknown
aFHostIdentity"Identity(1      ??9      ??A      ??I      ??aN?i?????????Unknown?
?GHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??aN?i1----????Unknown
THHostMul"Mul(1      ??9      ??A      ??I      ??aN?i?????????Unknown
|IHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aN?iA<<<<????Unknown
vJHostMul"%binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aN?i?????????Unknown
}KHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aN?iQKKKK????Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??aN?i?????????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aN?iaZZZZ????Unknown
yNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aN?i?????????Unknown
?OHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aN?iqiiii????Unknown
?PHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??aN?i?????????Unknown
?QHostReadVariableOp"-sequential_59/dense_183/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aN?i?xxxx????Unknown
?RHostReadVariableOp"-sequential_59/dense_184/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aN?i     ???Unknown
WSHostReluGrad".gradient_tape/sequential_59/dense_183/ReluGrad(i     ???Unknown2CPU