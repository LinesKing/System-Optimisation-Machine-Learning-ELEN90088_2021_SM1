"?h
BHostIDLE"IDLE1     ??@A     ??@a{????i{?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     X?@9     X?@A     X?@I     X?@a:e2Mdo??i&a!?????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      A@9      A@A      A@I      A@a?#??m?i?D?%???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      <@9      <@A      <@I      <@a?YU??Zh?i ??in=???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      ;@9      ;@A      ;@I      ;@a)??8|g?i)^???T???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?@@9     ?@@A      ;@I      ;@a)??8|g?iR?M?fl???Unknown
iHostWriteSummary"WriteSummary(1      8@9      8@A      8@I      8@aC?$s2?d?i??G????Unknown?
yHost_FusedMatMul"sequential_83/dense_258/BiasAdd(1      8@9      8@A      8@I      8@aC?$s2?d?i~04@'????Unknown
`	HostGatherV2"
GatherV2_1(1      *@9      *@A      *@I      *@a?????V?i??2v????Unknown
?
HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      (@9      (@A      (@I      (@aC?$s2?T?iE!l?????Unknown
dHostDataset"Iterator::Model(1     ?F@9     ?F@A      &@I      &@ah4???"S?i_???w????Unknown
gHostStridedSlice"strided_slice(1      &@9      &@A      &@I      &@ah4???"S?iy?U?????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@aeᶬKPO?i1?@?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aeᶬKPO?i?,?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      "@9      "@A      "@I      "@aeᶬKPO?i?61?????Unknown
?HostMatMul".gradient_tape/sequential_83/dense_259/MatMul_1(1      "@9      "@A      "@I      "@aeᶬKPO?iYdDY????Unknown
?HostMatMul",gradient_tape/sequential_83/dense_260/MatMul(1      "@9      "@A      "@I      "@aeᶬKPO?i??V-????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a?????K?i???"????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?????K?iU:#????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?????K?i??`?????Unknown
?HostMatMul",gradient_tape/sequential_83/dense_258/MatMul(1       @9       @A       @I       @a?????K?i-?????Unknown
yHost_FusedMatMul"sequential_83/dense_259/BiasAdd(1       @9       @A       @I       @a?????K?i?y?U????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a?YU??ZH?i
???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?YU??ZH?i`?p?%???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?YU??ZH?i?9ҁ<???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?YU??ZH?i?3;S!???Unknown
?HostMatMul",gradient_tape/sequential_83/dense_259/MatMul(1      @9      @A      @I      @a?YU??ZH?ibd??i'???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aC?$s2?D?i?-2?,???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aC?$s2?D?i????1???Unknown?
VHostMean"Mean(1      @9      @A      @I      @aC?$s2?D?iԿk7???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aC?$s2?D?i??'J<???Unknown
~ HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aC?$s2?D?i R?3?A???Unknown
?!HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aC?$s2?D?iFB@?F???Unknown
?"HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aC?$s2?D?il??L?K???Unknown
?#Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aC?$s2?D?i??{Y*Q???Unknown
?$HostBiasAddGrad"9gradient_tape/sequential_83/dense_260/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aC?$s2?D?i?vfbV???Unknown
t%HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a???_eA?i?s?ŻZ???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a???_eA?i?p?%_???Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???_eA?i?m??nc???Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???_eA?i?jx??g???Unknown
z)HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???_eA?i?gPE!l???Unknown
?*HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a???_eA?ivd(?zp???Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_83/dense_259/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???_eA?ika ?t???Unknown
V,HostCast"Cast(1      @9      @A      @I      @a?????;?i/??Nx???Unknown
?-HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????;?i??&k?{???Unknown
?.HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????;?i??9D???Unknown
?/HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?????;?i{$MѾ????Unknown
v0HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?????;?i?U`?9????Unknown
v1HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?????;?i?s7?????Unknown
}2HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      @9      @A      @I      @a?????;?iǶ??.????Unknown
`3HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?????;?i?癝?????Unknown
u4HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?????;?iO?P$????Unknown
~5HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?????;?iI??????Unknown
?6HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?????;?i?yӶ????Unknown
?7HostBiasAddGrad"9gradient_tape/sequential_83/dense_258/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?????;?i???i?????Unknown
n8HostTanh"sequential_83/dense_258/Tanh(1      @9      @A      @I      @a?????;?i_??????Unknown
?9HostReadVariableOp".sequential_83/dense_259/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?????;?i#Љ????Unknown
y:Host_FusedMatMul"sequential_83/dense_260/BiasAdd(1      @9      @A      @I      @a?????;?i?< ?????Unknown
t;HostSigmoid"sequential_83/dense_260/Sigmoid(1      @9      @A      @I      @a?????;?i?m36????Unknown
X<HostCast"Cast_3(1      @9      @A      @I      @aC?$s2?4?i>ҁ<????Unknown
\=HostGreater"Greater(1      @9      @A      @I      @aC?$s2?4?i?6?B?????Unknown
u>HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aC?$s2?4?id?IS????Unknown
r?HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @aC?$s2?4?i??lO?????Unknown
?@HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aC?$s2?4?i?d?U?????Unknown
bAHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aC?$s2?4?i?	\'????Unknown
?BHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @aC?$s2?4?i?-Xbþ???Unknown
xCHostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @aC?$s2?4?iC??h_????Unknown
?DHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      @9      @A      @I      @aC?$s2?4?i???n?????Unknown
?EHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      @9      @A      @I      @aC?$s2?4?ii[Cu?????Unknown
?FHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aC?$s2?4?i???{3????Unknown
?GHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      @9      @A      @I      @aC?$s2?4?i?$???????Unknown
?HHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @aC?$s2?4?i"?.?k????Unknown
~IHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @aC?$s2?4?i??|?????Unknown
?JHostTanhGrad".gradient_tape/sequential_83/dense_259/TanhGrad(1      @9      @A      @I      @aC?$s2?4?iHR˔?????Unknown
?KHostReadVariableOp"-sequential_83/dense_259/MatMul/ReadVariableOp(1      @9      @A      @I      @aC?$s2?4?i۶??????Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?????+?i=O???????Unknown
vMHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?????+?i??,N?????Unknown
XNHostCast"Cast_5(1       @9       @A       @I       @a?????+?i???w????Unknown
XOHostEqual"Equal(1       @9       @A       @I       @a?????+?ic@5????Unknown
TPHostMul"Mul(1       @9       @A       @I       @a?????+?iŰ?Z?????Unknown
|QHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?????+?i'IS??????Unknown
dRHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?????+?i???m????Unknown
vSHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?????+?i?yfg*????Unknown
vTHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?????+?iM???????Unknown
wUHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?????+?i??y?????Unknown
?VHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?????+?iCtb????Unknown
?WHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?????+?isی?????Unknown
?XHostTanhGrad".gradient_tape/sequential_83/dense_258/TanhGrad(1       @9       @A       @I       @a?????+?i?s'?????Unknown
?YHostMatMul".gradient_tape/sequential_83/dense_260/MatMul_1(1       @9       @A       @I       @a?????+?i7???????Unknown
?ZHostReadVariableOp".sequential_83/dense_258/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?????+?i??)?W????Unknown
?[HostReadVariableOp"-sequential_83/dense_258/MatMul/ReadVariableOp(1       @9       @A       @I       @a?????+?i?<?3????Unknown
?\HostReadVariableOp"-sequential_83/dense_260/MatMul/ReadVariableOp(1       @9       @A       @I       @a?????+?i]?<??????Unknown
X]HostCast"Cast_4(1      ??9      ??A      ??I      ??a??????i??:?????Unknown
a^HostIdentity"Identity(1      ??9      ??A      ??I      ??a??????i?m???????Unknown?
s_HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a??????i?9??n????Unknown
j`HostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a??????i!P@M????Unknown
waHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??????iR??+????Unknown
ybHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??????i??ٙ
????Unknown
?cHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??????i?j?F?????Unknown
?dHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??????i?6c??????Unknown
?eHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??????i(??????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??????iG??L?????Unknown
?gHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??????ix???c????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??????i?gv?B????Unknown
niHostTanh"sequential_83/dense_259/Tanh(1      ??9      ??A      ??I      ??a??????i?3;S!????Unknown
?jHostReadVariableOp".sequential_83/dense_260/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??????i     ???Unknown*?g
uHostFlushSummaryWriter"FlushSummaryWriter(1     X?@9     X?@A     X?@I     X?@a??????i???????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      A@9      A@A      A@I      A@a)???????i!({_????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      <@9      <@A      <@I      <@a@bw?#??i333333???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      ;@9      ;@A      ;@I      ;@a?U???Y??i?=ɚ?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?@@9     ?@@A      ;@I      ;@a?U???Y??i?H_̈???Unknown
iHostWriteSummary"WriteSummary(1      8@9      8@A      8@I      8@a?/fD???i
R?%? ???Unknown?
yHost_FusedMatMul"sequential_83/dense_258/BiasAdd(1      8@9      8@A      8@I      8@a?/fD???i?[?Hp????Unknown
`HostGatherV2"
GatherV2_1(1      *@9      *@A      *@I      *@a?H?n???i?`yƬ
???Unknown
?	HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      (@9      (@A      (@I      (@a?/fD???ijeؕV???Unknown
d
HostDataset"Iterator::Model(1     ?F@9     ?F@A      &@I      &@aW?]ie??i?i?}+????Unknown
gHostStridedSlice"strided_slice(1      &@9      &@A      &@I      &@aW?]ie??in?"?????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@av??fw|?i?q/?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@av??fw|?i:ua??S???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      "@9      "@A      "@I      "@av??fw|?i?x???????Unknown
?HostMatMul".gradient_tape/sequential_83/dense_259/MatMul_1(1      "@9      "@A      "@I      "@av??fw|?iX|?W|????Unknown
?HostMatMul",gradient_tape/sequential_83/dense_260/MatMul(1      "@9      "@A      "@I      "@av??fw|?i??$k????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a۔??My?i??1???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a۔??My?i;???c???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a۔??My?ie?'H=????Unknown
?HostMatMul",gradient_tape/sequential_83/dense_258/MatMul(1       @9       @A       @I       @a۔??My?i??7??????Unknown
yHost_FusedMatMul"sequential_83/dense_259/BiasAdd(1       @9       @A       @I       @a۔??My?i??G
t????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a@bw?#v?i~?5??'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a@bw?#v?iB?#?T???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a@bw?#v?i??K????Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a@bw?#v?iʚ?ݓ????Unknown
?HostMatMul",gradient_tape/sequential_83/dense_259/MatMul(1      @9      @A      @I      @a@bw?#v?i?????????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?/fD?r?iퟹ[?????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?/fD?r?iL????$???Unknown?
VHostMean"Mean(1      @9      @A      @I      @a?/fD?r?i??Qm?J???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?/fD?r?i
???p???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?/fD?r?ii??~?????Unknown
? HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?/fD?r?iȫ??????Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?/fD?r?i'????????Unknown
?"Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?/fD?r?i??M????Unknown
?#HostBiasAddGrad"9gradient_tape/sequential_83/dense_260/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?/fD?r?i???t.???Unknown
t$HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a???o?iߴþN???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a???o?iٶm۶m???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???o?iӸ?W????Unknown
?'HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???o?iͺ??????Unknown
z(HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???o?iǼk1?????Unknown
?)HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a???o?i??N;????Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_83/dense_259/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???o?i???j????Unknown
V+HostCast"Cast(1      @9      @A      @I      @a۔??Mi?iP?G*%???Unknown
?,HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a۔??Mi?i????w>???Unknown
?-HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a۔??Mi?iz?W|?W???Unknown
?.HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a۔??Mi?i??,q???Unknown
v/HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a۔??Mi?i??g?`????Unknown
v0HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a۔??Mi?i9?????Unknown
}1HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      @9      @A      @I      @a۔??Mi?i??w>?????Unknown
`2HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a۔??Mi?ic???I????Unknown
u3HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a۔??Mi?i?·??????Unknown
~4HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a۔??Mi?i??P????Unknown
?5HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a۔??Mi?i"җ 3"???Unknown
?6HostBiasAddGrad"9gradient_tape/sequential_83/dense_258/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a۔??Mi?i????;???Unknown
n7HostTanh"sequential_83/dense_258/Tanh(1      @9      @A      @I      @a۔??Mi?iLէa?T???Unknown
?8HostReadVariableOp".sequential_83/dense_259/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a۔??Mi?i??/n???Unknown
y9Host_FusedMatMul"sequential_83/dense_260/BiasAdd(1      @9      @A      @I      @a۔??Mi?ivط?i????Unknown
t:HostSigmoid"sequential_83/dense_260/Sigmoid(1      @9      @A      @I      @a۔??Mi?i??s?????Unknown
X;HostCast"Cast_3(1      @9      @A      @I      @a?/fD?b?i;ۥ??????Unknown
\<HostGreater"Greater(1      @9      @A      @I      @a?/fD?b?ik???????Unknown
u=HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?/fD?b?i??q@?????Unknown
r>HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?/fD?b?i??ׄ?????Unknown
??HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a?/fD?b?i??=ɚ????Unknown
b@HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?/fD?b?i+??????Unknown
?AHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a?/fD?b?i[?	R?%???Unknown
xBHostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @a?/fD?b?i??o??8???Unknown
?CHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      @9      @A      @I      @a?/fD?b?i???ڃK???Unknown
?DHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      @9      @A      @I      @a?/fD?b?i??;~^???Unknown
?EHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?/fD?b?i??cxq???Unknown
?FHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      @9      @A      @I      @a?/fD?b?iK??r????Unknown
?GHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?/fD?b?i{?m?l????Unknown
~HHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a?/fD?b?i???0g????Unknown
?IHostTanhGrad".gradient_tape/sequential_83/dense_259/TanhGrad(1      @9      @A      @I      @a?/fD?b?i??9ua????Unknown
?JHostReadVariableOp"-sequential_83/dense_259/MatMul/ReadVariableOp(1      @9      @A      @I      @a?/fD?b?iퟹ[????Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a۔??MY?i????????Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a۔??MY?i??'j?????Unknown
XMHostCast"Cast_5(1       @9       @A       @I       @a۔??MY?ii?kBP????Unknown
XNHostEqual"Equal(1       @9       @A       @I       @a۔??MY?i3??????Unknown
TOHostMul"Mul(1       @9       @A       @I       @a۔??MY?i????????Unknown
|PHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a۔??MY?i??7?D???Unknown
dQHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a۔??MY?i??{??(???Unknown
vRHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a۔??MY?i[??{?5???Unknown
vSHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a۔??MY?i%?T9B???Unknown
wTHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a۔??MY?i??G,?N???Unknown
?UHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a۔??MY?i????[???Unknown
?VHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a۔??MY?i????-h???Unknown
?WHostTanhGrad".gradient_tape/sequential_83/dense_258/TanhGrad(1       @9       @A       @I       @a۔??MY?iM???t???Unknown
?XHostMatMul".gradient_tape/sequential_83/dense_260/MatMul_1(1       @9       @A       @I       @a۔??MY?i?W?{????Unknown
?YHostReadVariableOp".sequential_83/dense_258/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a۔??MY?i???e"????Unknown
?ZHostReadVariableOp"-sequential_83/dense_258/MatMul/ReadVariableOp(1       @9       @A       @I       @a۔??MY?i???=ɚ???Unknown
?[HostReadVariableOp"-sequential_83/dense_260/MatMul/ReadVariableOp(1       @9       @A       @I       @a۔??MY?iu?#p????Unknown
X\HostCast"Cast_4(1      ??9      ??A      ??I      ??a۔??MI?i??E?í???Unknown
a]HostIdentity"Identity(1      ??9      ??A      ??I      ??a۔??MI?i??g?????Unknown?
s^HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a۔??MI?i???Zj????Unknown
j_HostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a۔??MI?i	??ƽ????Unknown
w`HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a۔??MI?in??2????Unknown
yaHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a۔??MI?i????d????Unknown
?bHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a۔??MI?i8??????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a۔??MI?i??3w????Unknown
?dHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a۔??MI?i?U?^????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a۔??MI?ig?wO?????Unknown
?fHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a۔??MI?i????????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a۔??MI?i1??'Y????Unknown
nhHostTanh"sequential_83/dense_259/Tanh(1      ??9      ??A      ??I      ??a۔??MI?i??ݓ?????Unknown
?iHostReadVariableOp".sequential_83/dense_260/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a۔??MI?i?????????Unknown2CPU