"?[
BHostIDLE"IDLE1     ??@A     ??@a?O$?Ҽ??i?O$?Ҽ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@aL? &W??i?e??????Unknown?
uHost_FusedMatMul"sequential_26/dense_86/Relu(1      C@9      C@A      C@I      C@aE?)͋?q?i1???\A???Unknown
iHostWriteSummary"WriteSummary(1      =@9      =@A      =@I      =@ai^???Sj?i???[???Unknown?
?HostGreaterEqual"8gradient_tape/kl_divergence/clip_by_value_1/GreaterEqual(1      8@9      8@A      8@I      8@aW?+??e?i?H8?yq???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      6@9      6@A      6@I      6@aP$?Ҽ?c?i??
br????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?@@9     ?@@A      3@I      3@aE?)͋?a?i????????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@aAL? &W`?iѼ?	????Unknown
g	HostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@aAL? &W`?ik:`????Unknown
?
HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      0@9      0@A      0@I      0@at?n??]?i}"???????Unknown
?HostReadVariableOp"-sequential_26/dense_86/BiasAdd/ReadVariableOp(1      0@9      0@A      0@I      0@at?n??]?i?٨?l????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      .@9      .@A      .@I      .@am?w6?;[?i???
????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ,@9      @A      ,@I      @af???kY?i;?2t?????Unknown
?HostSelectV2"4gradient_tape/kl_divergence/clip_by_value_1/SelectV2(1      ,@9      ,@A      ,@I      ,@af???kY?i???v????Unknown
?HostMatMul"-gradient_tape/sequential_26/dense_87/MatMul_1(1      (@9      (@A      (@I      (@aW?+??U?i7`??Z???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aI8?y?'R?i?2t?n???Unknown
HostMatMul"+gradient_tape/sequential_26/dense_87/MatMul(1      $@9      $@A      $@I      $@aI8?y?'R?io1?????Unknown
dHostDataset"Iterator::Model(1      >@9      >@A       @I       @at?n??M?i???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @at?n??M?iϼ?	'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @at?n??M?i?\AL.???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @af???kI?i??	?4???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @af???kI?i?X??;???Unknown
?HostDynamicStitch")gradient_tape/kl_divergence/DynamicStitch(1      @9      @A      @I      @af???kI?i-???\A???Unknown
HostMatMul"+gradient_tape/sequential_26/dense_88/MatMul(1      @9      @A      @I      @af???kI?ig:`?G???Unknown
uHost_FusedMatMul"sequential_26/dense_87/Relu(1      @9      @A      @I      @af???kI?i?y?'N???Unknown
xHost_FusedMatMul"sequential_26/dense_88/BiasAdd(1      @9      @A      @I      @af???kI?i?٨?lT???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aW?+??E?i???Q?Y???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aW?+??E?ic???Q_???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aW?+??E?i'???d???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aW?+??E?i?l?w6j???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_26/dense_86/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aW?+??E?i?Q?٨o???Unknown
? HostBiasAddGrad"8gradient_tape/sequential_26/dense_88/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aW?+??E?is6?;u???Unknown
?!HostCast"Nkl_divergence/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aW?+??E?i7???z???Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @aI8?y?'B?i??S????Unknown
?#Host	LessEqual"5gradient_tape/kl_divergence/clip_by_value_1/LessEqual(1      @9      @A      @I      @aI8?y?'B?i?????????Unknown
$HostMatMul"+gradient_tape/sequential_26/dense_86/MatMul(1      @9      @A      @I      @aI8?y?'B?i!W?+????Unknown
?%HostBiasAddGrad"8gradient_tape/sequential_26/dense_87/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aI8?y?'B?io?n??????Unknown
?&HostReadVariableOp",sequential_26/dense_87/MatMul/ReadVariableOp(1      @9      @A      @I      @aI8?y?'B?i?)͋?????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @at?n??=?i?"?????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @at?n??=?im1??????Unknown
V)HostMean"Mean(1      @9      @A      @I      @at?n??=?iE??O$????Unknown
?*HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??=?i???ş???Unknown
?+HostSelectV2"6gradient_tape/kl_divergence/clip_by_value_1/SelectV2_1(1      @9      @A      @I      @at?n??=?i??F}g????Unknown
?,HostTile"0gradient_tape/kl_divergence/weighted_loss/Tile_1(1      @9      @A      @I      @at?n??=?iͼ?	????Unknown
j-HostRealDiv"kl_divergence/truediv(1      @9      @A      @I      @at?n??=?i?????????Unknown
?.HostReadVariableOp"-sequential_26/dense_87/BiasAdd/ReadVariableOp(1      @9      @A      @I      @at?n??=?i}?\AL????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aW?+??5?i?
br????Unknown
V0HostCast"Cast(1      @9      @A      @I      @aW?+??5?iA}g??????Unknown
X1HostCast"Cast_3(1      @9      @A      @I      @aW?+??5?i??l?w????Unknown
X2HostEqual"Equal(1      @9      @A      @I      @aW?+??5?ibr1????Unknown
b3HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aW?+??5?ig?w6?????Unknown
?4HostMatMul"-gradient_tape/sequential_26/dense_88/MatMul_1(1      @9      @A      @I      @aW?+??5?i?F}g?????Unknown
b5HostLog"kl_divergence/Log(1      @9      @A      @I      @aW?+??5?i+???\????Unknown
p6HostMaximum"kl_divergence/clip_by_value(1      @9      @A      @I      @aW?+??5?i?+??????Unknown
x7HostMinimum"#kl_divergence/clip_by_value/Minimum(1      @9      @A      @I      @aW?+??5?i??????Unknown
b8HostMul"kl_divergence/mul(1      @9      @A      @I      @aW?+??5?iQ?+?????Unknown
p9HostSum"kl_divergence/weighted_loss/Sum(1      @9      @A      @I      @aW?+??5?i???\A????Unknown
?:HostReadVariableOp",sequential_26/dense_86/MatMul/ReadVariableOp(1      @9      @A      @I      @aW?+??5?i????????Unknown
?;HostReadVariableOp",sequential_26/dense_88/MatMul/ReadVariableOp(1      @9      @A      @I      @aW?+??5?iwg???????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @at?n??-?ic^???????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @at?n??-?iOUUUU????Unknown
X>HostCast"Cast_5(1       @9       @A       @I       @at?n??-?i;L? &????Unknown
T?HostMul"Mul(1       @9       @A       @I       @at?n??-?i'C??????Unknown
s@HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @at?n??-?i:`??????Unknown
uAHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @at?n??-?i?0???????Unknown
|BHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @at?n??-?i?'Ni????Unknown
`CHostDivNoNan"
div_no_nan(1       @9       @A       @I       @at?n??-?i?k:????Unknown
wDHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @at?n??-?i???
????Unknown
?EHost	ZerosLike"6gradient_tape/kl_divergence/clip_by_value_1/zeros_like(1       @9       @A       @I       @at?n??-?i???????Unknown
xFHostNeg"'gradient_tape/kl_divergence/truediv/Neg(1       @9       @A       @I       @at?n??-?i?v{?????Unknown
?GHostRealDiv"-gradient_tape/kl_divergence/truediv/RealDiv_1(1       @9       @A       @I       @at?n??-?i???F}????Unknown
?HHostDivNoNan":gradient_tape/kl_divergence/weighted_loss/value/div_no_nan(1       @9       @A       @I       @at?n??-?is?'N????Unknown
?IHostReluGrad"-gradient_tape/sequential_26/dense_86/ReluGrad(1       @9       @A       @I       @at?n??-?i_???????Unknown
rJHostMaximum"kl_divergence/clip_by_value_1(1       @9       @A       @I       @at?n??-?iK?٨?????Unknown
zKHostMinimum"%kl_divergence/clip_by_value_1/Minimum(1       @9       @A       @I       @at?n??-?i7?2t?????Unknown
wLHostDivNoNan"!kl_divergence/weighted_loss/value(1       @9       @A       @I       @at?n??-?i#͋??????Unknown
?MHostReadVariableOp"-sequential_26/dense_88/BiasAdd/ReadVariableOp(1       @9       @A       @I       @at?n??-?i??
b????Unknown
sNHostSigmoid"sequential_26/dense_88/Sigmoid(1       @9       @A       @I       @at?n??-?i??=?2????Unknown
XOHostCast"Cast_4(1      ??9      ??A      ??I      ??at?n???iq6?;????Unknown
uPHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??at?n???i籖?????Unknown
wQHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??at?n???i]-C?????Unknown
yRHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??at?n???iӨ?l?????Unknown
?SHostBroadcastTo"'gradient_tape/kl_divergence/BroadcastTo(1      ??9      ??A      ??I      ??at?n???iI$?Ҽ????Unknown
~THost
Reciprocal"&gradient_tape/kl_divergence/Reciprocal(1      ??9      ??A      ??I      ??at?n???i??H8?????Unknown
pUHostMul"gradient_tape/kl_divergence/mul(1      ??9      ??A      ??I      ??at?n???i5???????Unknown
tVHostMul"#gradient_tape/kl_divergence/mul/Mul(1      ??9      ??A      ??I      ??at?n???i???v????Unknown
tWHostSum"#gradient_tape/kl_divergence/mul/Sum(1      ??9      ??A      ??I      ??at?n???i!Ni^????Unknown
?XHostRealDiv"-gradient_tape/kl_divergence/truediv/RealDiv_2(1      ??9      ??A      ??I      ??at?n???i????F????Unknown
xYHostMul"'gradient_tape/kl_divergence/truediv/mul(1      ??9      ??A      ??I      ??at?n???i	?4/????Unknown
?ZHostReluGrad"-gradient_tape/sequential_26/dense_87/ReluGrad(1      ??9      ??A      ??I      ??at?n???i??S?????Unknown
?[HostSigmoidGrad"8gradient_tape/sequential_26/dense_88/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??at?n???i?????????Unknown
b\HostSum"kl_divergence/Sum(1      ??9      ??A      ??I      ??at?n???i?=?2t ???Unknown
]HostCast"-kl_divergence/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??at?n???is{?e? ???Unknown
4^HostIdentity"Identity(is{?e? ???Unknown?
b_Host	ZerosLike"8gradient_tape/kl_divergence/clip_by_value_1/zeros_like_1(is{?e? ???Unknown
M`HostSum")gradient_tape/kl_divergence/truediv/Sum_1(is{?e? ???Unknown*?Z
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@aw?qG???iw?qG????Unknown?
uHost_FusedMatMul"sequential_26/dense_86/Relu(1      C@9      C@A      C@I      C@a
Dp??(??i?????????Unknown
iHostWriteSummary"WriteSummary(1      =@9      =@A      =@I      =@a+?Ӄ???i?fl%????Unknown?
?HostGreaterEqual"8gradient_tape/kl_divergence/clip_by_value_1/GreaterEqual(1      8@9      8@A      8@I      8@a"8vi??iw?qG???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      6@9      6@A      6@I      6@a??t????i7`+!????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?@@9     ?@@A      3@I      3@a
Dp??(??iW???g^???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@a?3T1???ivi??????Unknown
gHostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@a?3T1???i?5eMYS???Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      0@9      0@A      0@I      0@a??JH7??i???n6????Unknown
?
HostReadVariableOp"-sequential_26/dense_86/BiasAdd/ReadVariableOp(1      0@9      0@A      0@I      0@a??JH7??i????-???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      .@9      .@A      .@I      .@aa?*?Ӄ??i????"????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ,@9      @A      ,@I      @a?}A_Ї?i???[d????Unknown
?HostSelectV2"4gradient_tape/kl_divergence/clip_by_value_1/SelectV2(1      ,@9      ,@A      ,@I      ,@a?}A_Ї?i??إQ???Unknown
?HostMatMul"-gradient_tape/sequential_26/dense_87/MatMul_1(1      (@9      (@A      (@I      (@a"8vi??i)??K????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aA?.???i:,|?U????Unknown
HostMatMul"+gradient_tape/sequential_26/dense_87/MatMul(1      $@9      $@A      $@I      $@aA?.???iKH7`+???Unknown
dHostDataset"Iterator::Model(1      >@9      >@A       @I       @a??JH7{?iY?̫?a???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a??JH7{?ig?b<=????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a??JH7{?iuX?̫????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?}A_?w?i?R{?L????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?}A_?w?i?L?I?-???Unknown
?HostDynamicStitch")gradient_tape/kl_divergence/DynamicStitch(1      @9      @A      @I      @a?}A_?w?i?F??]???Unknown
HostMatMul"+gradient_tape/sequential_26/dense_88/MatMul(1      @9      @A      @I      @a?}A_?w?i?@?.????Unknown
uHost_FusedMatMul"sequential_26/dense_87/Relu(1      @9      @A      @I      @a?}A_?w?i?:??ϼ???Unknown
xHost_FusedMatMul"sequential_26/dense_88/BiasAdd(1      @9      @A      @I      @a?}A_?w?i?4
Dp????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a"8vit?i?xz0C???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a"8vit?iѼ?>???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a"8vit?i? [	?f???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a"8vit?i?D???????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_26/dense_86/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a"8vit?i??;⎸???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_26/dense_88/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a"8vit?i?̫?a????Unknown
? HostCast"Nkl_divergence/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a"8vit?i?4
???Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @aA?.?q?i?y?9,???Unknown
?"Host	LessEqual"5gradient_tape/kl_divergence/clip_by_value_1/LessEqual(1      @9      @A      @I      @aA?.?q?i-??>N???Unknown
#HostMatMul"+gradient_tape/sequential_26/dense_86/MatMul(1      @9      @A      @I      @aA?.?q?i?4
Dp???Unknown
?$HostBiasAddGrad"8gradient_tape/sequential_26/dense_87/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aA?.?q?i'I?$I????Unknown
?%HostReadVariableOp",sequential_26/dense_87/MatMul/ReadVariableOp(1      @9      @A      @I      @aA?.?q?i0??>N????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??JH7k?i7?:??????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a??JH7k?i>??ϼ????Unknown
V(HostMean"Mean(1      @9      @A      @I      @a??JH7k?iE_?????Unknown
?)HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??JH7k?iL7`+!???Unknown
?*HostSelectV2"6gradient_tape/kl_divergence/clip_by_value_1/SelectV2_1(1      @9      @A      @I      @a??JH7k?iSf?b<???Unknown
?+HostTile"0gradient_tape/kl_divergence/weighted_loss/Tile_1(1      @9      @A      @I      @a??JH7k?iZ????W???Unknown
j,HostRealDiv"kl_divergence/truediv(1      @9      @A      @I      @a??JH7k?ia??8?r???Unknown
?-HostReadVariableOp"-sequential_26/dense_87/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??JH7k?ih?F?????Unknown
t.HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a"8vid?im?~?q????Unknown
V/HostCast"Cast(1      @9      @A      @I      @a"8vid?ir۶m۶???Unknown
X0HostCast"Cast_3(1      @9      @A      @I      @a"8vid?iw???D????Unknown
X1HostEqual"Equal(1      @9      @A      @I      @a"8vid?i|'Z?????Unknown
b2HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a"8vid?i?A_?????Unknown
?3HostMatMul"-gradient_tape/sequential_26/dense_88/MatMul_1(1      @9      @A      @I      @a"8vid?i?c?F????Unknown
b4HostLog"kl_divergence/Log(1      @9      @A      @I      @a"8vid?i??ϼ????Unknown
p5HostMaximum"kl_divergence/clip_by_value(1      @9      @A      @I      @a"8vid?i??3T1???Unknown
x6HostMinimum"#kl_divergence/clip_by_value/Minimum(1      @9      @A      @I      @a"8vid?i?????E???Unknown
b7HostMul"kl_divergence/mul(1      @9      @A      @I      @a"8vid?i??w'Z???Unknown
p8HostSum"kl_divergence/weighted_loss/Sum(1      @9      @A      @I      @a"8vid?i????n???Unknown
?9HostReadVariableOp",sequential_26/dense_86/MatMul/ReadVariableOp(1      @9      @A      @I      @a"8vid?i?/??????Unknown
?:HostReadVariableOp",sequential_26/dense_88/MatMul/ReadVariableOp(1      @9      @A      @I      @a"8vid?i?Q ?c????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??JH7[?i??E&?????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??JH7[?i?)kʚ????Unknown
X=HostCast"Cast_5(1       @9       @A       @I       @a??JH7[?i???n6????Unknown
T>HostMul"Mul(1       @9       @A       @I       @a??JH7[?i???????Unknown
s?HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??JH7[?i?m۶m????Unknown
u@HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??JH7[?i?? [	????Unknown
|AHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??JH7[?i?E&??????Unknown
`BHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??JH7[?i??K?@???Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??JH7[?i?qG????Unknown
?DHost	ZerosLike"6gradient_tape/kl_divergence/clip_by_value_1/zeros_like(1       @9       @A       @I       @a??JH7[?iǉ??w???Unknown
xEHostNeg"'gradient_tape/kl_divergence/truediv/Neg(1       @9       @A       @I       @a??JH7[?i????-???Unknown
?FHostRealDiv"-gradient_tape/kl_divergence/truediv/RealDiv_1(1       @9       @A       @I       @a??JH7[?i?a?3?:???Unknown
?GHostDivNoNan":gradient_tape/kl_divergence/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??JH7[?i???JH???Unknown
?HHostReluGrad"-gradient_tape/sequential_26/dense_86/ReluGrad(1       @9       @A       @I       @a??JH7[?i?9,|?U???Unknown
rIHostMaximum"kl_divergence/clip_by_value_1(1       @9       @A       @I       @a??JH7[?i֥Q ?c???Unknown
zJHostMinimum"%kl_divergence/clip_by_value_1/Minimum(1       @9       @A       @I       @a??JH7[?i?w?q???Unknown
wKHostDivNoNan"!kl_divergence/weighted_loss/value(1       @9       @A       @I       @a??JH7[?i?}?h?~???Unknown
?LHostReadVariableOp"-sequential_26/dense_88/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??JH7[?i???U????Unknown
sMHostSigmoid"sequential_26/dense_88/Sigmoid(1       @9       @A       @I       @a??JH7[?i?U???????Unknown
XNHostCast"Cast_4(1      ??9      ??A      ??I      ??a??JH7K?i????????Unknown
uOHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??JH7K?i??U?????Unknown
wPHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??JH7K?i?w'Z????Unknown
yQHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??JH7K?i?-2?'????Unknown
?RHostBroadcastTo"'gradient_tape/kl_divergence/BroadcastTo(1      ??9      ??A      ??I      ??a??JH7K?i??D??????Unknown
~SHost
Reciprocal"&gradient_tape/kl_divergence/Reciprocal(1      ??9      ??A      ??I      ??a??JH7K?i??W??????Unknown
pTHostMul"gradient_tape/kl_divergence/mul(1      ??9      ??A      ??I      ??a??JH7K?i?Ojo?????Unknown
tUHostMul"#gradient_tape/kl_divergence/mul/Mul(1      ??9      ??A      ??I      ??a??JH7K?i?}A_????Unknown
tVHostSum"#gradient_tape/kl_divergence/mul/Sum(1      ??9      ??A      ??I      ??a??JH7K?i???-????Unknown
?WHostRealDiv"-gradient_tape/kl_divergence/truediv/RealDiv_2(1      ??9      ??A      ??I      ??a??JH7K?i?q???????Unknown
xXHostMul"'gradient_tape/kl_divergence/truediv/mul(1      ??9      ??A      ??I      ??a??JH7K?i?'???????Unknown
?YHostReluGrad"-gradient_tape/sequential_26/dense_87/ReluGrad(1      ??9      ??A      ??I      ??a??JH7K?i??ǉ?????Unknown
?ZHostSigmoidGrad"8gradient_tape/sequential_26/dense_88/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a??JH7K?i???[d????Unknown
b[HostSum"kl_divergence/Sum(1      ??9      ??A      ??I      ??a??JH7K?i?I?-2????Unknown
\HostCast"-kl_divergence/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??JH7K?i      ???Unknown
4]HostIdentity"Identity(i      ???Unknown?
b^Host	ZerosLike"8gradient_tape/kl_divergence/clip_by_value_1/zeros_like_1(i      ???Unknown
M_HostSum")gradient_tape/kl_divergence/truediv/Sum_1(i      ???Unknown2CPU