"?e
BHostIDLE"IDLE1     ??@A     ??@aFb?|=??iFb?|=???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?7YJ??iB6?e????Unknown?
dHostDataset"Iterator::Model(1     ?G@9     ?G@A     ?G@I     ?G@a?	\w?r?iUFeTFe???Unknown
XHostCast"Cast_5(1     ?D@9     ?D@A     ?D@I     ?D@aZu5,zp?i@_Ϭ:????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      B@9      B@A      @@I      @@a?-&?v?i?in??#?????Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@ah?h?f?i?ۋt????Unknown?
vHost_FusedMatMul"sequential_47/dense_146/Relu(1      9@9      9@A      9@I      9@a?Ӆ?d?i??ʨ?????Unknown
XHostEqual"Equal(1      5@9      5@A      5@I      5@a??`?i?m۶m????Unknown
?	HostMatMul",gradient_tape/sequential_47/dense_147/MatMul(1      2@9      2@A      2@I      2@a???Ӆ?\?i2S?y?????Unknown
?
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      1@I      1@a??xd?S[?iz??x?????Unknown
?HostReadVariableOp".sequential_47/dense_147/BiasAdd/ReadVariableOp(1      .@9      .@A      .@I      .@a??Ӆ?X?i_y??????Unknown
?HostMatMul".gradient_tape/sequential_47/dense_147/MatMul_1(1      &@9      &@A      &@I      &@a???ѮQ?i??Yu???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?-&?v?I?i
\w????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?-&?v?I?i?Q?Q???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?-&?v?I?i ?ֲ????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a?-&?v?I?i????-&???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?-&?v?I?i6.Q??,???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ah?h?F?i??VH<2???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ah?h?F?i?n\??7???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ah?h?F?iDb?|=???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @ah?h?F?i??gVC???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @ah?h?F?i?Om??H???Unknown
?HostMatMul",gradient_tape/sequential_47/dense_148/MatMul(1      @9      @A      @I      @ah?h?F?iR?r
^N???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a\??7YJC?i{???0S???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a\??7YJC?i??7X???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a\??7YJC?i??\??\???Unknown
?HostMatMul",gradient_tape/sequential_47/dense_146/MatMul(1      @9      @A      @I      @a\??7YJC?i?̪c?a???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??7YJ@?i?A6?e???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??7YJ@?i?h??i???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a??7YJ@?i۶m۶m???Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??7YJ@?i???q???Unknown
? HostBiasAddGrad"9gradient_tape/sequential_47/dense_147/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??7YJ@?i?R???u???Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_47/dense_148/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??7YJ@?i??0S?y???Unknown
V"HostCast"Cast(1      @9      @A      @I      @a?-&?v?9?i?Eb?|???Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?-&?v?9?iL??p3????Unknown
V$HostMean"Mean(1      @9      @A      @I      @a?-&?v?9?i??j????Unknown
u%HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?-&?v?9?i?3???????Unknown
?&HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?-&?v?9?i?؉?؉???Unknown
z'HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?-&?v?9?id}h?????Unknown
v(HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?-&?v?9?i*"G?F????Unknown
v)HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?-&?v?9?i??%?}????Unknown
v*HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?-&?v?9?i?kٴ????Unknown
~+HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?-&?v?9?i|???????Unknown
?,HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?-&?v?9?iB???"????Unknown
?-HostBiasAddGrad"9gradient_tape/sequential_47/dense_146/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?-&?v?9?iZ?Z????Unknown
?.HostMatMul".gradient_tape/sequential_47/dense_148/MatMul_1(1      @9      @A      @I      @a?-&?v?9?i??~?????Unknown
y/Host_FusedMatMul"sequential_47/dense_148/BiasAdd(1      @9      @A      @I      @a?-&?v?9?i??]#Ȧ???Unknown
?0HostReadVariableOp".sequential_47/dense_148/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?-&?v?9?iZH<2?????Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a\??7YJ3?i?Cc}h????Unknown
X2HostCast"Cast_3(1      @9      @A      @I      @a\??7YJ3?i????Ѯ???Unknown
\3HostGreater"Greater(1      @9      @A      @I      @a\??7YJ3?i;?;????Unknown
s4HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a\??7YJ3?i?6?^?????Unknown
?5HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a\??7YJ3?i>2??????Unknown
r6HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a\??7YJ3?i?-&?v????Unknown
?7HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a\??7YJ3?if)M@?????Unknown
~8HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a\??7YJ3?i?$t?I????Unknown
`9HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a\??7YJ3?i? ?ֲ????Unknown
?:HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a\??7YJ3?i"?!????Unknown
?;HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a\??7YJ3?i??l?????Unknown
v<Host_FusedMatMul"sequential_47/dense_147/Relu(1      @9      @A      @I      @a\??7YJ3?iJ??????Unknown
t=HostSigmoid"sequential_47/dense_148/Sigmoid(1      @9      @A      @I      @a\??7YJ3?i?7X????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?-&?v?)?iAa???????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?-&?v?)?i???????Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?-&?v?)?i??*????Unknown
dAHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?-&?v?)?ijX? ?????Unknown
jBHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?-&?v?)?iͪc?a????Unknown
vCHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?-&?v?)?i0??/?????Unknown
vDHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?-&?v?)?i?OB??????Unknown
?EHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?-&?v?)?i???>4????Unknown
}FHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?-&?v?)?iY? ??????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?-&?v?)?i?F?Mk????Unknown
?HHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?-&?v?)?i???????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?-&?v?)?i??n\?????Unknown
?JHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?-&?v?)?i?=??=????Unknown
?KHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?-&?v?)?iH?Mk?????Unknown
?LHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?-&?v?)?i????t????Unknown
?MHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a?-&?v?)?i5,z????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?-&?v?)?iq???????Unknown
?OHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1       @9       @A       @I       @a?-&?v?)?i??
?G????Unknown
?PHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?-&?v?)?i7,z?????Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?-&?v?)?i?~??~????Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?-&?v?)?i??X????Unknown
?SHostReluGrad".gradient_tape/sequential_47/dense_146/ReluGrad(1       @9       @A       @I       @a?-&?v?)?i`#Ȧ?????Unknown
?THostReluGrad".gradient_tape/sequential_47/dense_147/ReluGrad(1       @9       @A       @I       @a?-&?v?)?i?u7.Q????Unknown
?UHostReadVariableOp".sequential_47/dense_146/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?-&?v?)?i&Ȧ??????Unknown
?VHostReadVariableOp"-sequential_47/dense_146/MatMul/ReadVariableOp(1       @9       @A       @I       @a?-&?v?)?i?=?????Unknown
?WHostReadVariableOp"-sequential_47/dense_147/MatMul/ReadVariableOp(1       @9       @A       @I       @a?-&?v?)?i?l??#????Unknown
?XHostReadVariableOp"-sequential_47/dense_148/MatMul/ReadVariableOp(1       @9       @A       @I       @a?-&?v?)?iO??K?????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?-&?v??i?h??????Unknown
XZHostCast"Cast_4(1      ??9      ??A      ??I      ??a?-&?v??i?d?Z????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a?-&?v??i???(????Unknown?
T\HostMul"Mul(1      ??9      ??A      ??I      ??a?-&?v??id?Z?????Unknown
u]HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?-&?v??iD??????Unknown
b^HostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a?-&?v??iu?B??????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?-&?v??i?_??_????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?-&?v??i??i-????Unknown
?aHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?-&?v??i?i-?????Unknown
?bHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      ??9      ??A      ??I      ??a?-&?v??i9[!??????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?-&?v??ijٴ?????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?-&?v??i???xd????Unknown
?eHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?-&?v??i?VH<2????Unknown
~fHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a?-&?v??i?????????Unknown
JgHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown*?d
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a??!?8??i??!?8???Unknown?
dHostDataset"Iterator::Model(1     ?G@9     ?G@A     ?G@I     ?G@a??,?'???i?b?_?6???Unknown
XHostCast"Cast_5(1     ?D@9     ?D@A     ?D@I     ?D@a:??|????i?ҋC???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      B@9      B@A      @@I      @@a??#?a???i?/[?.????Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@aS??????i?*?H|X???Unknown?
vHost_FusedMatMul"sequential_47/dense_146/Relu(1      9@9      9@A      9@I      9@a??c?????iMJ??????Unknown
XHostEqual"Equal(1      5@9      5@A      5@I      5@a??~??^??i?FH/Q???Unknown
?HostMatMul",gradient_tape/sequential_47/dense_147/MatMul(1      2@9      2@A      2@I      2@a?FH/Q??iXghR????Unknown
?	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      1@I      1@au	68???i~?FH/???Unknown
?
HostReadVariableOp".sequential_47/dense_147/BiasAdd/ReadVariableOp(1      .@9      .@A      .@I      .@a??ҋC??i???w=_???Unknown
?HostMatMul".gradient_tape/sequential_47/dense_147/MatMul_1(1      &@9      &@A      &@I      &@aj9??f?}?i1??D?????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??#?a?u?i????????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a??#?a?u?ic7??#????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??#?a?u?i?~??^???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a??#?a?u?i??nT?G???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a??#?a?u?i.Q?r???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aS????r?i????????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aS????r?iz?z????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aS????r?i 
?ZN????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aS????r?i???!
???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aS????r?ilO1?/???Unknown
?HostMatMul",gradient_tape/sequential_47/dense_148/MatMul(1      @9      @A      @I      @aS????r?i???U???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a???t	6p?iŻ??4v???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???t	6p?ixq? ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a???t	6p?i+'r?????Unknown
?HostMatMul",gradient_tape/sequential_47/dense_146/MatMul(1      @9      @A      @I      @a???t	6p?i??[?x????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aտlm?k?i?Iɢ}????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aտlm?k?i^?6]????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aտlm?k?i#??(???Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aտlm?k?iޏҋC???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_47/dense_147/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aտlm?k?i??~??^???Unknown
? HostBiasAddGrad"9gradient_tape/sequential_47/dense_148/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aտlm?k?i^i?F?y???Unknown
V!HostCast"Cast(1      @9      @A      @I      @a??#?a?e?i+?ݨ2????Unknown
?"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??#?a?e?i???
Ф???Unknown
V#HostMean"Mean(1      @9      @A      @I      @a??#?a?e?i?Կlm????Unknown
u$HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??#?a?e?i????
????Unknown
?%HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??#?a?e?i_?0?????Unknown
z&HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??#?a?e?i,@??E????Unknown
v'HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??#?a?e?i?c??????Unknown
v(HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??#?a?e?iƇuV?&???Unknown
v)HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??#?a?e?i??f?<???Unknown
~*HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??#?a?e?i`?W?Q???Unknown
?+HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??#?a?e?i-?H|Xg???Unknown
?,HostBiasAddGrad"9gradient_tape/sequential_47/dense_146/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??#?a?e?i?:??|???Unknown
?-HostMatMul".gradient_tape/sequential_47/dense_148/MatMul_1(1      @9      @A      @I      @a??#?a?e?i?:+@?????Unknown
y.Host_FusedMatMul"sequential_47/dense_148/BiasAdd(1      @9      @A      @I      @a??#?a?e?i?^?0????Unknown
?/HostReadVariableOp".sequential_47/dense_148/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??#?a?e?ia?ν???Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a???t	6`?i:]?????Unknown
X1HostCast"Cast_3(1      @9      @A      @I      @a???t	6`?i8?:????Unknown
\2HostGreater"Greater(1      @9      @A      @I      @a???t	6`?i?l p????Unknown
s3HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a???t	6`?i???)?????Unknown
?4HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???t	6`?i??U3????Unknown
r5HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a???t	6`?i|??<???Unknown
?6HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a???t	6`?iV~?FH/???Unknown
~7HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a???t	6`?i0Y?O~????Unknown
`8HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a???t	6`?i
4)Y?O???Unknown
?9HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???t	6`?i??b?_???Unknown
?:HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a???t	6`?i??l p???Unknown
v;Host_FusedMatMul"sequential_47/dense_147/Relu(1      @9      @A      @I      @a???t	6`?i?ćuV????Unknown
t<HostSigmoid"sequential_47/dense_148/Sigmoid(1      @9      @A      @I      @a???t	6`?ir??~?????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??#?a?U?iX1?/[????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a??#?a?U?i>???)????Unknown
|?HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??#?a?U?i$U???????Unknown
d@HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??#?a?U?i
??Bǻ???Unknown
jAHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??#?a?U?i?x???????Unknown
vBHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??#?a?U?i?
Фd????Unknown
vCHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??#?a?U?i???U3????Unknown
?DHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??#?a?U?i?.?????Unknown
}EHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??#?a?U?i?????????Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?inR?h?????Unknown
?GHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??#?a?U?iT??n???Unknown
xHHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??#?a?U?i:v??<???Unknown
?IHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??#?a?U?i ?{???Unknown
?JHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??#?a?U?i??,?'???Unknown
?KHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??#?a?U?i?+?ݨ2???Unknown
?LHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a??#?a?U?iҽ??w=???Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??#?a?U?i?O~?FH???Unknown
?NHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1       @9       @A       @I       @a??#?a?U?i??v?S???Unknown
?OHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??#?a?U?i?so??]???Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??#?a?U?ijhR?h???Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??#?a?U?iP?`?s???Unknown
?RHostReluGrad".gradient_tape/sequential_47/dense_146/ReluGrad(1       @9       @A       @I       @a??#?a?U?i6)Y?O~???Unknown
?SHostReluGrad".gradient_tape/sequential_47/dense_147/ReluGrad(1       @9       @A       @I       @a??#?a?U?i?Qe????Unknown
?THostReadVariableOp".sequential_47/dense_146/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?iMJ?????Unknown
?UHostReadVariableOp"-sequential_47/dense_146/MatMul/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i??Bǻ????Unknown
?VHostReadVariableOp"-sequential_47/dense_147/MatMul/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i?p;x?????Unknown
?WHostReadVariableOp"-sequential_47/dense_148/MatMul/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i?4)Y????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??#?a?E?i?K???????Unknown
XYHostCast"Cast_4(1      ??9      ??A      ??I      ??a??#?a?E?i??,?'????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a??#?a?E?i?ݨ2?????Unknown?
T[HostMul"Mul(1      ??9      ??A      ??I      ??a??#?a?E?i?&%??????Unknown
u\HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??#?a?E?iso??]????Unknown
b]HostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a??#?a?E?if?<?????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??#?a?E?iY??,????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??#?a?E?iLJ??????Unknown
?`Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??#?a?E?i???E?????Unknown
?aHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      ??9      ??A      ??I      ??a??#?a?E?i2??b????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??#?a?E?i%%???????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??#?a?E?inO1????Unknown
?dHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??#?a?E?i????????Unknown
~eHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a??#?a?E?i?????????Unknown
JfHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown2CPU