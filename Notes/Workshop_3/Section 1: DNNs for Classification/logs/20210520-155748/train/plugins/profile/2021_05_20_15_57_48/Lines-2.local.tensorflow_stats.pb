"?d
BHostIDLE"IDLE1     ??@A     ??@a?F{^????i?F{^?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     @R@9     @R@A     @R@I     @R@aݴޙ?k??iJ<J3?#???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a????l~??iL???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?A@9     ?A@A     ?A@I     ?A@a????l~?i?????????Unknown?
dHostDataset"Iterator::Model(1     ?V@9     ?V@A      2@I      2@a?g?c2p?i??R?Y???Unknown
gHostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@a?g?c2p?i??P?!???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      0@9      0@A      0@I      0@a:nc;\?l?i??P??>???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      ,@I      ,@as ???1i?i????W???Unknown
?	HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@as ???1i?i?鸍?p???Unknown
o
Host_FusedMatMul"sequential/dense/Relu(1      *@9      *@A      *@I      *@a??@?:eg?i?*??R????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      *@9      *@A      *@I      *@a??@?:eg?i?k??????Unknown
\HostGreater"Greater(1      (@9      (@A      (@I      (@a???,??e?i?ňP????Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      (@9      (@A      (@I      (@a???,??e?i????????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a?[?h??c?i	U[ݴ????Unknown?
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@a?[?h??c?ie)Ĭ?????Unknown
`HostDivNoNan"
div_no_nan(1      $@9      $@A      $@I      $@a?$??a?i?Gi????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?g?c2`?ix?J*????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?g?c2`?if,??$???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a:nc;\?\?i?I<J3???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a:nc;\?\?i?zg??A???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @a:nc;\?\?i?,??P???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @as ???1Y?i(ߐ?\???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @as ???1Y?i?#9?Gi???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @as ???1Y?i???u???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @as ???1Y?i??yy????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @as ???1Y?iGr????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @as ???1Y?i??j?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a???,??U?i?V7?w????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???,??U?i???C????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a???,??U?if?c2????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?$??Q?ixp6?????Unknown?
? HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?$??Q?i??L????Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?$??Q?i????????Unknown
?"HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?$??Q?i??e????Unknown
v#HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?$??Q?i????????Unknown
?$HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?$??Q?i?;S????Unknown
?%HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?$??Q?i??%????Unknown
?&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?$??Q?i?Y?????Unknown
t'Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a?$??Q?i??%???Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a:nc;\?L?i????>???Unknown
?)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a:nc;\?L?i????q???Unknown
?*HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a:nc;\?L?i?s???"???Unknown
v+HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a:nc;\?L?ixL??)???Unknown
~,HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a:nc;\?L?iT%Y
1???Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a:nc;\?L?i0?#0=8???Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a:nc;\?L?i?2p????Unknown
}/HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a:nc;\?L?i??AޢF???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???,??E?i?Ҍ?L???Unknown
V1HostMean"Mean(1      @9      @A      @I      @a???,??E?i2?? oQ???Unknown
s2HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a???,??E?i?#B?V???Unknown
z3HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???,??E?i|:nc;\???Unknown
v4HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???,??E?i!]???a???Unknown
b5HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a???,??E?i??g???Unknown
y6HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      @9      @A      @I      @a???,??E?ik?O?ml???Unknown
?7Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a???,??E?iŚ??q???Unknown
?8HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???,??E?i???	:w???Unknown
?9HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???,??E?iZ
1+?|???Unknown
?:HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a???,??E?i?,|L????Unknown
?;HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a???,??E?i?O?ml????Unknown
t<HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a:nc;\?<?i?N?????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a:nc;\?<?i?(?D?????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a:nc;\?<?i??]?8????Unknown
V?HostCast"Cast(1       @9       @A       @I       @a:nc;\?<?i\?ҕ???Unknown
X@HostCast"Cast_3(1       @9       @A       @I       @a:nc;\?<?i?ml?k????Unknown
XAHostEqual"Equal(1       @9       @A       @I       @a:nc;\?<?i8???????Unknown
uBHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a:nc;\?<?i?F{^?????Unknown
dCHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a:nc;\?<?i??7????Unknown
jDHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a:nc;\?<?i??5ѧ???Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a:nc;\?<?i???j????Unknown
?FHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a:nc;\?<?i^??????Unknown
vGHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a:nc;\?<?i?d x?????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a:nc;\?<?i:ѧ?6????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a:nc;\?<?i?=/Oй???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a:nc;\?<?i???i????Unknown
xKHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a:nc;\?<?i?>&????Unknown
?LHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a:nc;\?<?i??ő?????Unknown
?MHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a:nc;\?<?i`?L?5????Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a:nc;\?<?i?[?h?????Unknown
?OHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a:nc;\?<?i<?[?h????Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a:nc;\?<?i?4??????Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a:nc;\?<?i?j??????Unknown
}RHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a:nc;\?<?i??5????Unknown
?SHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a:nc;\?<?i?yy??????Unknown
oTHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @a:nc;\?<?ib? ?g????Unknown
XUHostCast"Cast_4(1      ??9      ??A      ??I      ??a:nc;\?,?i??ģ4????Unknown
XVHostCast"Cast_5(1      ??9      ??A      ??I      ??a:nc;\?,?i?R?Y????Unknown
TWHostMul"Mul(1      ??9      ??A      ??I      ??a:nc;\?,?i	L?????Unknown
|XHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a:nc;\?,?i>?Ś????Unknown
vYHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a:nc;\?,?iuu?zg????Unknown
}ZHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a:nc;\?,?i?+?04????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a:nc;\?,?i??Z? ????Unknown
?\HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a:nc;\?,?i???????Unknown
?]HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a:nc;\?,?iQN?Q?????Unknown
?^Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a:nc;\?,?i??g????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a:nc;\?,?i??i?3????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a:nc;\?,?i?p-s ????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a:nc;\?,?i-'?(?????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a:nc;\?,?idݴޙ????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a:nc;\?,?i??x?f????Unknown
~dHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a:nc;\?,?i?I<J3????Unknown
eHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a:nc;\?,?i     ???Unknown
?fHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a:nc;\?,?i??Z? ???Unknown
4gHostIdentity"Identity(i??Z? ???Unknown?
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i??Z? ???Unknown*?d
sHostDataset"Iterator::Model::ParallelMapV2(1     @R@9     @R@A     @R@I     @R@a????-???i????-????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a?as?ü?i???ca???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?A@9     ?A@A     ?A@I     ?A@a?as?ì?i?$I?$I???Unknown?
dHostDataset"Iterator::Model(1     ?V@9     ?V@A      2@I      2@a???????iB?P?"???Unknown
gHostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@a???????i????????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      0@9      0@A      0@I      0@a7& nL??iU????????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      ,@I      ,@ap?\???il???????Unknown
?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@ap?\???i?@M?????Unknown
o	Host_FusedMatMul"sequential/dense/Relu(1      *@9      *@A      *@I      *@a?
z^??it????????Unknown
q
Host_FusedMatMul"sequential/dense_1/Relu(1      *@9      *@A      *@I      *@a?
z^??ie??+?,???Unknown
\HostGreater"Greater(1      (@9      (@A      (@I      (@a??	?R???i04U_h???Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      (@9      (@A      (@I      (@a??	?R???i?я~?????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@aF
	????iP1???b???Unknown?
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@aF
	????i?y??B????Unknown
`HostDivNoNan"
div_no_nan(1      $@9      $@A      $@I      $@a?7??o??ia?H!?v???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a???????i???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a???????i?1i qc???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a7& nL??iRf鸢????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a7& nL??i??iq?5???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @a7& nL??i???)????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ap?\???i??Y?????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ap?\???i?+?,W???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @ap?\???i?Y:?(????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @ap?\???i???/4???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @ap?\???i????k???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @ap?\???i???2K????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a??	?R???i?|0???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??	?R???i?2K?e???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a??	?R???iZ??????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?7??o??i?z?$?????Unknown?
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?7??o??i??K8y7???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?7??o??i???K8y???Unknown
?!HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?7??o??i???^?????Unknown
v"HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?7??o??ia?;r?????Unknown
?#HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?7??o??iA??u>???Unknown
?$HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?7??o??i!@ܘ4????Unknown
?%HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?7??o??ia,??????Unknown
t&Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a?7??o??i??|?????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a7& nLz?i-???K8???Unknown
?(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a7& nLz?iy??w?l???Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a7& nLz?i??<T}????Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a7& nLz?i?|0????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a7& nLz?i]??
???Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a7& nLz?i???G????Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a7& nLz?i?9=??s???Unknown
}.HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a7& nLz?iAT}?y????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??	?R?s?i?g?F?????Unknown
V0HostMean"Mean(1      @9      @A      @I      @a??	?R?s?i?{??^????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?il??????Unknown
z2HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??	?R?s?i%?=6DF???Unknown
v3HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??	?R?s?i޶m۶m???Unknown
b4HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??	?R?s?i?ʝ?)????Unknown
y5HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      @9      @A      @I      @a??	?R?s?iP??%?????Unknown
?6Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??	?R?s?i	???????Unknown
?7HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??	?R?s?i?.p????Unknown
?8HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?i{^?2???Unknown
?9HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?i4-??fZ???Unknown
?:HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?i?@?_ف???Unknown
t;HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a7& nLj?iN??%????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a7& nLj?i9[?;r????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a7& nLj?i_h??????Unknown
V>HostCast"Cast(1       @9       @A       @I       @a7& nLj?i?u>????Unknown
X?HostCast"Cast_3(1       @9       @A       @I       @a7& nLj?i??^?W???Unknown
X@HostEqual"Equal(1       @9       @A       @I       @a7& nLj?iя~?????Unknown
uAHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i???b?9???Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a7& nLj?i???<T???Unknown
jCHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a7& nLj?iC??>?n???Unknown
rDHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a7& nLj?ii???Ո???Unknown
?EHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a7& nLj?i??"????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a7& nLj?i??>?n????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a7& nLj?i??^??????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i?~e????Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i'??S???Unknown
xJHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a7& nLj?iM?A?&???Unknown
?KHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a7& nLj?is ߯?@???Unknown
?LHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a7& nLj?i?-?9[???Unknown
?MHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a7& nLj?i?:??u???Unknown
?NHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a7& nLj?i?G??я???Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a7& nLj?iU_h????Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a7& nLj?i1b?j????Unknown
}QHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a7& nLj?iWo?D?????Unknown
?RHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i}|??????Unknown
oSHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @a7& nLj?i??? P???Unknown
XTHostCast"Cast_4(1      ??9      ??A      ??I      ??a7& nLZ?i6??Wv ???Unknown
XUHostCast"Cast_5(1      ??9      ??A      ??I      ??a7& nLZ?iɖ???-???Unknown
TVHostMul"Mul(1      ??9      ??A      ??I      ??a7& nLZ?i\???:???Unknown
|WHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a7& nLZ?i????G???Unknown
vXHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a7& nLZ?i??/4U???Unknown
}YHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a7& nLZ?i??k5b???Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7& nLZ?i??O?[o???Unknown
?[HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a7& nLZ?i;?_ف|???Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a7& nLZ?i??o?????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a7& nLZ?ia?GΖ???Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a7& nLZ?i?я~?????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a7& nLZ?i?؟?????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a7& nLZ?i߯?@????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a7& nLZ?i???#g????Unknown
?bHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a7& nLZ?i@??Z?????Unknown
~cHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a7& nLZ?i??ߑ?????Unknown
dHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a7& nLZ?if????????Unknown
?eHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a7& nLZ?i?????????Unknown
4fHostIdentity"Identity(i?????????Unknown?
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU