"?e
BHostIDLE"IDLE1     ??@A     ??@aPzz????iPzz?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     P?@9     P?@A     P?@I     P?@aMb??4Ź?ig|80???Unknown?
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@a&pݶ Ak?i?Y?q2???Unknown?
wHost_FusedMatMul"sequential_3/dense_10/BiasAdd(1      ?@9      ?@A      ?@I      ?@a&pݶ Ak?iG7??M???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      9@I      9@a@??z??e?i?? ??c???Unknown
`HostGatherV2"
GatherV2_1(1      3@9      3@A      3@I      3@aY??>:?`?i?q_?`t???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      0@9      0@A      0@I      0@a̔?@"\?i????q????Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      0@9      0@A      0@I      0@a̔?@"\?i'V???????Unknown
?	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@aK?,?_Z?iM????????Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      0@9      0@A      .@I      .@aK?,?_Z?is,???????Unknown
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1      .@9      .@A      .@I      .@aK?,?_Z?i????????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      .@9      .@A      .@I      .@aK?,?_Z?i???B????Unknown
rHostSigmoid"sequential_3/dense_12/Sigmoid(1      .@9      .@A      .@I      .@aK?,?_Z?i?m?r????Unknown
vHostSub"%binary_crossentropy/logistic_loss/sub(1      ,@9      ,@A      ,@I      ,@a3?ϝX?i?ќ??????Unknown
wHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ,@9      ,@A      ,@I      ,@a3?ϝX?i?5)?????Unknown
VHostCast"Cast(1      *@9      *@A      *@I      *@a渹??V?iÒ+}~????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      (@9      (@A      (@I      (@a?o???U?i{??C???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a ݎ?J?Q?i?/??	???Unknown
?HostMatMul",gradient_tape/sequential_3/dense_11/MatMul_1(1      $@9      $@A      $@I      $@a ݎ?J?Q?iXwl?????Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a ݎ?J?Q?iƾ?3k???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a̔?@"L?i?? ?s"???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a̔?@"L?i1q<|)???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @a̔?@"L?i5j???0???Unknown
~HostMatMul"*gradient_tape/sequential_3/dense_11/MatMul(1       @9       @A       @I       @a̔?@"L?iZ?E?7???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a3?ϝH?i[?׸?=???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a3?ϝH?i\?,?C???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a3?ϝH?i]9d?J???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a3?ϝH?i^k*+P???Unknown
~HostMatMul"*gradient_tape/sequential_3/dense_12/MatMul(1      @9      @A      @I      @a3?ϝH?i_???RV???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?o???E?i;?,??[???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?o???E?i?hN?`???Unknown?
~ HostMatMul"*gradient_tape/sequential_3/dense_10/MatMul(1      @9      @A      @I      @a?o???E?i???%f???Unknown
w!Host_FusedMatMul"sequential_3/dense_11/BiasAdd(1      @9      @A      @I      @a?o???E?i?H?lk???Unknown
w"Host_FusedMatMul"sequential_3/dense_12/BiasAdd(1      @9      @A      @I      @a?o???E?i?sx?p???Unknown
X#HostEqual"Equal(1      @9      @A      @I      @a ݎ?J?A?ib???u???Unknown
?$HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a ݎ?J?A?i??}y???Unknown
?%HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a ݎ?J?A?i??3p?}???Unknown
?&HostBiasAddGrad"7gradient_tape/sequential_3/dense_11/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ݎ?J?A?i???G????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a̔?@"<?i̅???Unknown
X(HostCast"Cast_3(1      @9      @A      @I      @a̔?@"<?i?;6GP????Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a̔?@"<?i@X^?Ԍ???Unknown
?*HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a̔?@"<?i?t??X????Unknown
?+HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a̔?@"<?if??ݓ???Unknown
?,HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a̔?@"<?i???Oa????Unknown
?-HostBiasAddGrad"7gradient_tape/sequential_3/dense_10/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a̔?@"<?i?????????Unknown
?.HostBiasAddGrad"7gradient_tape/sequential_3/dense_12/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a̔?@"<?i?&?i????Unknown
?/HostMatMul",gradient_tape/sequential_3/dense_12/MatMul_1(1      @9      @A      @I      @a̔?@"<?i?O?????Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?o???5?i ?G?????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?o???5?i?.?y4????Unknown
X2HostCast"Cast_4(1      @9      @A      @I      @a?o???5?i?C)?ש???Unknown
\3HostGreater"Greater(1      @9      @A      @I      @a?o???5?ijY??z????Unknown
|4HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a?o???5?i?ne????Unknown
V5HostSum"Sum_2(1      @9      @A      @I      @a?o???5?iF?@?????Unknown
j6HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?o???5?i???qd????Unknown
?7HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?o???5?i"???????Unknown
v8HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?o???5?i???Ԫ????Unknown
v9HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?o???5?i??{N????Unknown
?:HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?o???5?il?8?????Unknown
?;HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?o???5?i??i?????Unknown
?<HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?o???5?iHV?7????Unknown
?=Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?o???5?i?/???????Unknown
?>HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?o???5?i$E??}????Unknown
??HostReadVariableOp",sequential_3/dense_10/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?o???5?i?Z00!????Unknown
l@HostTanh"sequential_3/dense_10/Tanh(1      @9      @A      @I      @a?o???5?i p?a?????Unknown
?AHostReadVariableOp",sequential_3/dense_11/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?o???5?in?l?g????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a̔?@",?i????)????Unknown
XCHostCast"Cast_5(1       @9       @A       @I       @a̔?@",?i ????????Unknown
VDHostMean"Mean(1       @9       @A       @I       @a̔?@",?iI????????Unknown
uEHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a̔?@",?i???p????Unknown
dFHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a̔?@",?i???82????Unknown
rGHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a̔?@",?i$??Y?????Unknown
zHHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a̔?@",?im??z?????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a̔?@",?i???x????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a̔?@",?i?!?:????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a̔?@",?iH5??????Unknown
bLHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a̔?@",?i?"I??????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a̔?@",?i?0] ?????Unknown
~NHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a̔?@",?i#?qAC????Unknown
?OHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a̔?@",?ilM?b????Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a̔?@",?i?[???????Unknown
?QHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a̔?@",?i?i???????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a̔?@",?iGx??K????Unknown
?SHostReadVariableOp"+sequential_3/dense_10/MatMul/ReadVariableOp(1       @9       @A       @I       @a̔?@",?i????????Unknown
?THostReadVariableOp"+sequential_3/dense_12/MatMul/ReadVariableOp(1       @9       @A       @I       @a̔?@",?iٔ??????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a̔?@"?i??s?????Unknown
?VHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a̔?@"?i#??(?????Unknown
uWHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a̔?@"?iH??9s????Unknown
wXHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a̔?@"?im?JT????Unknown
yYHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a̔?@"?i???Z5????Unknown
?ZHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a̔?@"?i??%k????Unknown
?[HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a̔?@"?i?Ư{?????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a̔?@"?i?9??????Unknown
?]HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a̔?@"?i&?Ü?????Unknown
?^HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a̔?@"?iK?M??????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a̔?@"?ip?׽{????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a̔?@"?i??a?\????Unknown
?aHostTanhGrad",gradient_tape/sequential_3/dense_10/TanhGrad(1      ??9      ??A      ??I      ??a̔?@"?i????=????Unknown
?bHostTanhGrad",gradient_tape/sequential_3/dense_11/TanhGrad(1      ??9      ??A      ??I      ??a̔?@"?i??u?????Unknown
?cHostReadVariableOp"+sequential_3/dense_11/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a̔?@"?i     ???Unknown
ldHostTanh"sequential_3/dense_11/Tanh(1      ??9      ??A      ??I      ??a̔?@"?i?E?p ???Unknown
?eHostReadVariableOp",sequential_3/dense_12/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a̔?@"?i&?? ???Unknown
4fHostIdentity"Identity(i&?? ???Unknown?
'gHostMul"Mul(i&?? ???Unknown
WhHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i&?? ???Unknown
YiHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i&?? ???Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     P?@9     P?@A     P?@I     P?@a?ʄm??i?ʄm???Unknown?
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@a8?	? 8??i?袋.????Unknown?
wHost_FusedMatMul"sequential_3/dense_10/BiasAdd(1      ?@9      ?@A      ?@I      ?@a8?	? 8??id5{??c???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      9@I      9@a?b????i?I???????Unknown
`HostGatherV2"
GatherV2_1(1      3@9      3@A      3@I      3@aw????iU&l??T???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      0@9      0@A      0@I      0@a?и[???i?fO?}????Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      0@9      0@A      0@I      0@a?и[???i?2a???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@a?C?刄?iW?'?>V???Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      0@9      0@A      .@I      .@a?C?刄?i???b????Unknown
s
HostReadVariableOp"SGD/Cast/ReadVariableOp(1      .@9      .@A      .@I      .@a?C?刄?i??)?????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      .@9      .@A      .@I      .@a?C?刄?iM???L???Unknown
rHostSigmoid"sequential_3/dense_12/Sigmoid(1      .@9      .@A      .@I      .@a?C?刄?i???X͞???Unknown
vHostSub"%binary_crossentropy/logistic_loss/sub(1      ,@9      ,@A      ,@I      ,@a*?Ap*??i??w????Unknown
wHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ,@9      ,@A      ,@I      ,@a*?Ap*??i9?	? 8???Unknown
VHostCast"Cast(1      *@9      *@A      *@I      *@a?)??ˁ?i?9"?P???Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      (@9      (@A      (@I      (@an?ʄm??i©L?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aa?2a{?i???=?????Unknown
?HostMatMul",gradient_tape/sequential_3/dense_11/MatMul_1(1      $@9      $@A      $@I      $@aa?2a{?i0?袋.???Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@aa?2a{?ig?6Ne???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?и[?u?i?b??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?и[?u?i?w?????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @a?и[?u?i뢋.?????Unknown
~HostMatMul"*gradient_tape/sequential_3/dense_11/MatMul(1       @9       @A       @I       @a?и[?u?iC??????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a*?Ap*s?i=????:???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a*?Ap*s?ic?2a???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a*?Ap*s?i?????????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a*?Ap*s?i??
hܭ???Unknown
~HostMatMul"*gradient_tape/sequential_3/dense_12/MatMul(1      @9      @A      @I      @a*?Ap*s?i?_?H1????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @an?ʄmp?i??#R????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @an?ʄmp?iи[????Unknown?
~HostMatMul"*gradient_tape/sequential_3/dense_10/MatMul(1      @9      @A      @I      @an?ʄmp?i8Ne?6???Unknown
w Host_FusedMatMul"sequential_3/dense_11/BiasAdd(1      @9      @A      @I      @an?ʄmp?iY@?n?W???Unknown
w!Host_FusedMatMul"sequential_3/dense_12/BiasAdd(1      @9      @A      @I      @an?ʄmp?izxxxxx???Unknown
X"HostEqual"Equal(1      @9      @A      @I      @aa?2ak?i?|?ٓ???Unknown
?#HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aa?2ak?i????:????Unknown
?$HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aa?2ak?i˄m?????Unknown
?%HostBiasAddGrad"7gradient_tape/sequential_3/dense_11/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aa?2ak?i??C?????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?и[?e?i?X͞?????Unknown
X'HostCast"Cast_3(1      @9      @A      @I      @a?и[?e?i)??????Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?и[?e?i(?>V?'???Unknown
?)HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?и[?e?i>????=???Unknown
?*HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?и[?e?iT???S???Unknown
?+HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?и[?e?ijiiiii???Unknown
?,HostBiasAddGrad"7gradient_tape/sequential_3/dense_10/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?и[?e?i?9"?P???Unknown
?-HostBiasAddGrad"7gradient_tape/sequential_3/dense_12/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?и[?e?i?	? 8????Unknown
?.HostMatMul",gradient_tape/sequential_3/dense_12/MatMul_1(1      @9      @A      @I      @a?и[?e?i?ٓ|????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @an?ʄm`?i?u^?????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @an?ʄm`?i?)??????Unknown
X1HostCast"Cast_4(1      @9      @A      @I      @an?ʄm`?iܭ?
h????Unknown
\2HostGreater"Greater(1      @9      @A      @I      @an?ʄm`?i?I???????Unknown
|3HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @an?ʄm`?i???C????Unknown
V4HostSum"Sum_2(1      @9      @A      @I      @an?ʄm`?i?S?????Unknown
j5HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @an?ʄm`?i???Unknown
?6HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @an?ʄm`?i,?袋.???Unknown
v7HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @an?ʄm`?i<V?'?>???Unknown
v8HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @an?ʄm`?iL?}?fO???Unknown
?9HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @an?ʄm`?i\?H1?_???Unknown
?:HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @an?ʄm`?il*?Ap???Unknown
?;HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @an?ʄm`?i|??:?????Unknown
?<Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @an?ʄm`?i?b??????Unknown
?=HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @an?ʄm`?i??rD?????Unknown
?>HostReadVariableOp",sequential_3/dense_10/BiasAdd/ReadVariableOp(1      @9      @A      @I      @an?ʄm`?i??=??????Unknown
l?HostTanh"sequential_3/dense_10/Tanh(1      @9      @A      @I      @an?ʄm`?i?6Ne????Unknown
?@HostReadVariableOp",sequential_3/dense_11/BiasAdd/ReadVariableOp(1      @9      @A      @I      @an?ʄm`?i?????????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?и[?U?i?:???????Unknown
XBHostCast"Cast_5(1       @9       @A       @I       @a?и[?U?i⢋.?????Unknown
VCHostMean"Mean(1       @9       @A       @I       @a?и[?U?i?
hܭ????Unknown
uDHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a?и[?U?i?rD??????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?и[?U?i? 8?	???Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?и[?U?iC??????Unknown
zGHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?и[?U?i?ٓ|???Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?и[?U?i$?Ap*???Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?и[?U?i/{??c5???Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?и[?U?i:?n?W@???Unknown
bKHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?и[?U?iEKKKKK???Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?и[?U?iP?'?>V???Unknown
~MHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?и[?U?i[?2a???Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?и[?U?if??T&l???Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?и[?U?iq??w???Unknown
?PHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?и[?U?i|S??????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?и[?U?i??u^????Unknown
?RHostReadVariableOp"+sequential_3/dense_10/MatMul/ReadVariableOp(1       @9       @A       @I       @a?и[?U?i?#R?????Unknown
?SHostReadVariableOp"+sequential_3/dense_12/MatMul/ReadVariableOp(1       @9       @A       @I       @a?и[?U?i??.??????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?и[?E?i???b????Unknown
?UHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?и[?E?i??
hܭ???Unknown
uVHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?и[?E?i?'?>V????Unknown
wWHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?и[?E?i?[?и???Unknown
yXHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?и[?E?i????I????Unknown
?YHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?и[?E?i?????????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?и[?E?i????=????Unknown
?[Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?и[?E?i?+?q?????Unknown
?\HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?и[?E?i?_?H1????Unknown
?]HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?и[?E?iϓ|?????Unknown
?^HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?и[?E?i??j?$????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?и[?E?i??X͞????Unknown
?`HostTanhGrad",gradient_tape/sequential_3/dense_10/TanhGrad(1      ??9      ??A      ??I      ??a?и[?E?i?/G?????Unknown
?aHostTanhGrad",gradient_tape/sequential_3/dense_11/TanhGrad(1      ??9      ??A      ??I      ??a?и[?E?i?c5{?????Unknown
?bHostReadVariableOp"+sequential_3/dense_11/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?и[?E?i??#R????Unknown
lcHostTanh"sequential_3/dense_11/Tanh(1      ??9      ??A      ??I      ??a?и[?E?i??)?????Unknown
?dHostReadVariableOp",sequential_3/dense_12/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?и[?E?i?????????Unknown
4eHostIdentity"Identity(i?????????Unknown?
'fHostMul"Mul(i?????????Unknown
WgHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU