"?i
BHostIDLE"IDLE1     L?@A     L?@aN???_??iN???_???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@av?????i??k?Md???Unknown?
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@a???Nj?g?i?z??{???Unknown?
tHost_FusedMatMul"sequential_4/dense_13/Relu(1      7@9      7@A      7@I      7@a?????a?iomGy????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      9@9      9@A      5@I      5@a????`?i0?c?{????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@a,-X??\?i??????Unknown
^HostGatherV2"GatherV2(1      2@9      2@A      2@I      2@a\??Ir[?iG?<?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      1@I      1@a??Pw??Y?i???4?????Unknown
?	HostReadVariableOp"+sequential_4/dense_14/MatMul/ReadVariableOp(1      1@9      1@A      1@I      1@a??Pw??Y?i9{,?????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      .@9      .@A      .@I      .@a"?t?=?V?i^sF?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      .@9      .@A      .@I      .@a"?t?=?V?i??j|????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      .@9      .@A      .@I      .@a"?t?=?V?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      ,@9      ,@A      ,@I      ,@adV&?XU?i??z? ???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      (@9      (@A      (@I      (@a? *E1LR?i????	???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      (@9      (@A      (@I      (@a? *E1LR?i5?????Unknown
gHostStridedSlice"strided_slice(1      &@9      &@A      &@I      &@a*ֻ???P?i?r?G???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      $@9      $@A      $@I      $@a?V???~N?i̙QW?"???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a\??IrK?i?????)???Unknown
dHostDataset"Iterator::Model(1      ;@9      ;@A       @I       @a????eH?i7B?O?/???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a????eH?i??N??5???Unknown
?HostMatMul",gradient_tape/sequential_4/dense_14/MatMul_1(1       @9       @A       @I       @a????eH?i??<???Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_15/MatMul(1       @9       @A       @I       @a????eH?i8lҀ)B???Unknown
wHost_FusedMatMul"sequential_4/dense_15/BiasAdd(1       @9       @A       @I       @a????eH?i?$??BH???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @adV&?XE?iy???M???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @adV&?XE?i(?X?R???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @adV&?XE?i????EX???Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_14/MatMul(1      @9      @A      @I      @adV&?XE?i;+?ʛ]???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_4/dense_15/BiasAdd/BiasAddGrad(1      @9      @A      @I      @adV&?XE?iѬ??b???Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @a? *E1LB?iQ??g???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a? *E1LB?i?Afl???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a? *E1LB?iQ??(?p???Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a? *E1LB?i??5>u???Unknown
~!HostMatMul"*gradient_tape/sequential_4/dense_13/MatMul(1      @9      @A      @I      @a? *E1LB?iQ!ZA?y???Unknown
~"HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?V???~>?i?4? ?}???Unknown
?#HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?V???~>?i'H? q????Unknown
?$HostBiasAddGrad"7gradient_tape/sequential_4/dense_13/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?V???~>?i?[%?@????Unknown
t%Host_FusedMatMul"sequential_4/dense_14/Relu(1      @9      @A      @I      @a?V???~>?i?n??????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????e8?iRK?r????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a????e8?i?'?%*????Unknown
?(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a????e8?i?a?6????Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a????e8?iQ?A?C????Unknown
v*HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a????e8?i??">P????Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a????e8?i???\????Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a????e8?iPu??i????Unknown
?-HostBiasAddGrad"7gradient_tape/sequential_4/dense_14/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????e8?i?Q?Vv????Unknown
?.HostReadVariableOp"+sequential_4/dense_13/MatMul/ReadVariableOp(1      @9      @A      @I      @a????e8?i?-?	?????Unknown
?/HostReadVariableOp",sequential_4/dense_15/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????e8?iO
???????Unknown
]0HostCast"Adam/Cast_1(1      @9      @A      @I      @a? *E1L2?i???B٩???Unknown
v1HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a? *E1L2?i?T??"????Unknown
Y2HostPow"Adam/Pow(1      @9      @A      @I      @a? *E1L2?i? Ol????Unknown
[3HostPow"
Adam/Pow_1(1      @9      @A      @I      @a? *E1L2?iO?)յ????Unknown
X4HostCast"Cast_3(1      @9      @A      @I      @a? *E1L2?i?DR[?????Unknown
V5HostMean"Mean(1      @9      @A      @I      @a? *E1L2?i??z?H????Unknown
j6HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a? *E1L2?i??g?????Unknown
r7HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a? *E1L2?iO4??۹???Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a? *E1L2?i???s%????Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a? *E1L2?i?~?n????Unknown
v:HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a? *E1L2?i$F??????Unknown
?;HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a? *E1L2?iO?n????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a? *E1L2?i?n??K????Unknown
~=HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a? *E1L2?i???????Unknown
?>HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a? *E1L2?i????????Unknown
??Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a? *E1L2?iO^(????Unknown
?@HostMatMul",gradient_tape/sequential_4/dense_15/MatMul_1(1      @9      @A      @I      @a? *E1L2?i?:?q????Unknown
rAHostSigmoid"sequential_4/dense_15/Sigmoid(1      @9      @A      @I      @a? *E1L2?iϨb+?????Unknown
~BHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a????e(?i?ӄA????Unknown
vCHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a????e(?i%?C??????Unknown
eDHostAddN"Adam/gradients/AddN(1       @9       @A       @I       @a????e(?iP??7N????Unknown
tEHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a????e(?i{a$??????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a????e(?i?ϔ?Z????Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a????e(?i?=D?????Unknown
XHHostEqual"Equal(1       @9       @A       @I       @a????e(?i??u?g????Unknown
vIHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a????e(?i'???????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a????e(?iR?VPt????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a????e(?i}?Ʃ?????Unknown
?LHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a????e(?i?d7?????Unknown
?MHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a????e(?i?ҧ\????Unknown
?NHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a????e(?i?@??????Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a????e(?i)??????Unknown
?PHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a????e(?iT?h?????Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a????e(?i?i? ????Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a????e(?i????????Unknown
?SHostReluGrad",gradient_tape/sequential_4/dense_13/ReluGrad(1       @9       @A       @I       @a????e(?i?gJu-????Unknown
?THostReluGrad",gradient_tape/sequential_4/dense_14/ReluGrad(1       @9       @A       @I       @a????e(?i ֺγ????Unknown
tUHostReadVariableOp"Adam/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a????e?is?v????Unknown
oVHostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a????e?i*D+(:????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a????e?i?{?T?????Unknown
VXHostCast"Cast(1      ??9      ??A      ??I      ??a????e?iT????????Unknown
XYHostCast"Cast_4(1      ??9      ??A      ??I      ??a????e?ii?S??????Unknown
XZHostCast"Cast_5(1      ??9      ??A      ??I      ??a????e?i~ ?F????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a????e?i?W?
????Unknown?
T\HostMul"Mul(1      ??9      ??A      ??I      ??a????e?i??|4?????Unknown
v]HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a????e?i??4a?????Unknown
}^HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a????e?i????S????Unknown
`_HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a????e?i?3??????Unknown
w`HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????e?i?j]??????Unknown
?aHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a????e?i??????Unknown
xbHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a????e?i&??@`????Unknown
?cHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a????e?i;?m#????Unknown
?dHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a????e?iPG>??????Unknown
?eHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a????e?ie~?Ʃ????Unknown
?fHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a????e?iz???l????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a????e?i??f 0????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a????e?i?#M?????Unknown
~iHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a????e?i?Z?y?????Unknown
?jHostReadVariableOp",sequential_4/dense_13/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a????e?iΑ??y????Unknown
?kHostReadVariableOp",sequential_4/dense_14/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a????e?i??G?<????Unknown
?lHostReadVariableOp"+sequential_4/dense_15/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????e?i?????????Unknown
LmHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
YnHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?h
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@aK?D?0???iK?D?0????Unknown?
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@ak?J?͐?i?,u?`???Unknown?
tHost_FusedMatMul"sequential_4/dense_13/Relu(1      7@9      7@A      7@I      7@a~????i)q?^????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      9@9      9@A      5@I      5@ah????Ć?i???p???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@aR$ti???i'?ʇ?q???Unknown
^HostGatherV2"GatherV2(1      2@9      2@A      2@I      2@a?e??܃??i?????????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      1@I      1@a<??Pn??i[`?<?	???Unknown
?HostReadVariableOp"+sequential_4/dense_14/MatMul/ReadVariableOp(1      1@9      1@A      1@I      1@a<??Pn??i??~XS???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      .@9      .@A      .@I      .@a&*i?7C??i???^e????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      .@9      .@A      .@I      .@a&*i?7C??iJd?>r????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      .@9      .@A      .@I      .@a&*i?7C??i?~???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      ,@9      ,@A      ,@I      ,@a5?LiW[~?i??P?5S???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      (@9      (@A      (@I      (@a	?AZ&z?i[&@????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      (@9      (@A      (@I      (@a	?AZ&z?i??fJ????Unknown
gHostStridedSlice"strided_slice(1      &@9      &@A      &@I      &@a?_???w?i?"_??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      $@9      $@A      $@I      $@a??6K??u?i???l\???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?e??܃s?ig?|&d=???Unknown
dHostDataset"Iterator::Model(1      ;@9      ;@A       @I       @a??+<?Xq?i8K??`???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??+<?Xq?i	?m7ǂ???Unknown
?HostMatMul",gradient_tape/sequential_4/dense_14/MatMul_1(1       @9       @A       @I       @a??+<?Xq?i????x????Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_15/MatMul(1       @9       @A       @I       @a??+<?Xq?i?R^H*????Unknown
wHost_FusedMatMul"sequential_4/dense_15/BiasAdd(1       @9       @A       @I       @a??+<?Xq?i|????????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a5?LiW[n?iS??(7	???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a5?LiW[n?i*D??'???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a5?LiW[n?i???E???Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_14/MatMul(1      @9      @A      @I      @a5?LiW[n?i??{.Id???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_4/dense_15/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a5?LiW[n?i?*兤????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @a	?AZ&j?i?l???????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a	?AZ&j?ii??Ү????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a	?AZ&j?iF????????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a	?AZ&j?i#2N?????Unknown
~ HostMatMul"*gradient_tape/sequential_4/dense_13/MatMul(1      @9      @A      @I      @a	?AZ&j?i t?E????Unknown
~!HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??6K??e?i???:m???Unknown
?"HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??6K??e?i??>00???Unknown
?#HostBiasAddGrad"7gradient_tape/sequential_4/dense_13/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??6K??e?i??%?E???Unknown
t$Host_FusedMatMul"sequential_4/dense_14/Relu(1      @9      @A      @I      @a??6K??e?i?O?z[???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??+<?Xa?iu{??l???Unknown
\&HostGreater"Greater(1      @9      @A      @I      @a??+<?Xa?i^?M?+~???Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??+<?Xa?iGӉg?????Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a??+<?Xa?i0??+ݠ???Unknown
v)HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??+<?Xa?i+?5????Unknown
v*HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??+<?Xa?iW>??????Unknown
?+HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??+<?Xa?i??zx?????Unknown
?,HostBiasAddGrad"7gradient_tape/sequential_4/dense_14/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??+<?Xa?iԮ?<@????Unknown
?-HostReadVariableOp"+sequential_4/dense_13/MatMul/ReadVariableOp(1      @9      @A      @I      @a??+<?Xa?i??? ?????Unknown
?.HostReadVariableOp",sequential_4/dense_15/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??+<?Xa?i?/?????Unknown
]/HostCast"Adam/Cast_1(1      @9      @A      @I      @a	?AZ&Z?i?'\X????Unknown
v0HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a	?AZ&Z?i?H???"???Unknown
Y1HostPow"Adam/Pow(1      @9      @A      @I      @a	?AZ&Z?isi?~?/???Unknown
[2HostPow"
Adam/Pow_1(1      @9      @A      @I      @a	?AZ&Z?ib???<???Unknown
X3HostCast"Cast_3(1      @9      @A      @I      @a	?AZ&Z?iQ???I???Unknown
V4HostMean"Mean(1      @9      @A      @I      @a	?AZ&Z?i@?=8W???Unknown
j5HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a	?AZ&Z?i/?j?d???Unknown
r6HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a	?AZ&Z?i?^q???Unknown
?7HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a	?AZ&Z?i/??~???Unknown
z8HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a	?AZ&Z?i?O??????Unknown
v9HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a	?AZ&Z?i?p????Unknown
?:HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a	?AZ&Z?iڑL?????Unknown
b;HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a	?AZ&Z?iɲy>????Unknown
~<HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a	?AZ&Z?i?Ӧ?????Unknown
?=HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a	?AZ&Z?i???d????Unknown
?>Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a	?AZ&Z?i??????Unknown
??HostMatMul",gradient_tape/sequential_4/dense_15/MatMul_1(1      @9      @A      @I      @a	?AZ&Z?i?6.?????Unknown
r@HostSigmoid"sequential_4/dense_15/Sigmoid(1      @9      @A      @I      @a	?AZ&Z?itW[ ????Unknown
~AHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a??+<?XQ?ihmy??????Unknown
vBHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a??+<?XQ?i\???x???Unknown
eCHostAddN"Adam/gradients/AddN(1       @9       @A       @I       @a??+<?XQ?iP??D%???Unknown
tDHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??+<?XQ?iD?Ӧ????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??+<?XQ?i8??~???Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a??+<?XQ?i,?k*'???Unknown
XGHostEqual"Equal(1       @9       @A       @I       @a??+<?XQ?i ?-??/???Unknown
vHHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??+<?XQ?iL/?8???Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??+<?XQ?ij?/A???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??+<?XQ?i?2???I???Unknown
?KHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??+<?XQ?i?H?U?R???Unknown
?LHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??+<?XQ?i?^ķ4[???Unknown
?MHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a??+<?XQ?i?t??c???Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??+<?XQ?i̊ |?l???Unknown
?OHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??+<?XQ?i???9u???Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??+<?XQ?i??<@?}???Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??+<?XQ?i??Z??????Unknown
?RHostReluGrad",gradient_tape/sequential_4/dense_13/ReluGrad(1       @9       @A       @I       @a??+<?XQ?i??x?????Unknown
?SHostReluGrad",gradient_tape/sequential_4/dense_14/ReluGrad(1       @9       @A       @I       @a??+<?XQ?i???f?????Unknown
tTHostReadVariableOp"Adam/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a??+<?XA?i???A????Unknown
oUHostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a??+<?XA?i??ȗ????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??+<?XA?i~???????Unknown
VWHostCast"Cast(1      ??9      ??A      ??I      ??a??+<?XA?ix$?*D????Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a??+<?XA?ir/?[?????Unknown
XYHostCast"Cast_5(1      ??9      ??A      ??I      ??a??+<?XA?il:???????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a??+<?XA?ifE ?F????Unknown?
T[HostMul"Mul(1      ??9      ??A      ??I      ??a??+<?XA?i`P???Unknown
v\HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a??+<?XA?iZ[ ?????Unknown
}]HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??+<?XA?iTf-QI????Unknown
`^HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a??+<?XA?iNq<??????Unknown
w_HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??+<?XA?iH|K??????Unknown
?`HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a??+<?XA?iB?Z?K????Unknown
xaHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a??+<?XA?i<?i?????Unknown
?bHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??+<?XA?i6?xF?????Unknown
?cHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??+<?XA?i0??wN????Unknown
?dHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??+<?XA?i*????????Unknown
?eHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??+<?XA?i$????????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??+<?XA?iɴ
Q????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??+<?XA?i??;?????Unknown
~hHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a??+<?XA?i??l?????Unknown
?iHostReadVariableOp",sequential_4/dense_13/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??+<?XA?i???S????Unknown
?jHostReadVariableOp",sequential_4/dense_14/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??+<?XA?i??Ω????Unknown
?kHostReadVariableOp"+sequential_4/dense_15/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??+<?XA?i      ???Unknown
LlHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i      ???Unknown
YmHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i      ???Unknown2CPU