"?g
BHostIDLE"IDLE1     ??@A     ??@a8*??????i8*???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     H?@9     H?@A     H?@I     H?@a??a???i???j????Unknown?
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      E@9      E@A      E@I      E@a6???ajh?i?|?f?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      E@9      E@A     ?B@I     ?B@a?a+?I?e?i?V????Unknown
iHostWriteSummary"WriteSummary(1     ?B@9     ?B@A     ?B@I     ?B@a?a+?I?e?iPӌ??????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      ;@9      ;@A      4@I      4@a?쁬?@W?iF?Yy????Unknown
dHostAddN"SGD/gradients/AddN(1      3@9      3@A      3@I      3@a?:H?V?ic8???	???Unknown
vHost_FusedMatMul"sequential_68/dense_211/Relu(1      3@9      3@A      3@I      3@a?:H?V?i?\mu????Unknown
g	HostStridedSlice"strided_slice(1      3@9      3@A      3@I      3@a?:H?V?i??2????Unknown
s
HostDataset"Iterator::Model::ParallelMapV2(1      1@9      1@A      1@I      1@a8??E??S?ik??})???Unknown
tHostSigmoid"sequential_68/dense_213/Sigmoid(1      *@9      *@A      *@I      *@as?y?:N?i?3?1???Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@auQ???I?icp?q7???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      &@9      &@A      &@I      &@auQ???I?i?\???=???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?쁬?@G?i2}WĦC???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?쁬?@G?i???vI???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@aw?hx?D?iO?R?N???Unknown?
?HostMatMul",gradient_tape/sequential_68/dense_212/MatMul(1      "@9      "@A      "@I      "@aw?hx?D?i??6??S???Unknown
?HostMatMul".gradient_tape/sequential_68/dense_212/MatMul_1(1      "@9      "@A      "@I      "@aw?hx?D?i??P)Y???Unknown
?HostMatMul",gradient_tape/sequential_68/dense_213/MatMul(1      "@9      "@A      "@I      "@aw?hx?D?i5?jld^???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A       @I       @a?#?#2?B?i????
c???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      T@9      T@A       @I       @a?#?#2?B?i?y|??g???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?#?#2?B?i?`Xl???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?#?#2?B?iYG???p???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ay?'??F@?iI?Yu???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ay?'??F@?i9?}"y???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ay?'??F@?i)?u?3}???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @ay?'??F@?iom?E????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a??h5K?;?i0?s???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??h5K?;?iG?:]?????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a??h5K?;?i^v?F?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??h5K?;?iu#09????Unknown
? HostBiasAddGrad"9gradient_tape/sequential_68/dense_213/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??h5K?;?i??n?????Unknown
?!HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?쁬?@7?i?`D1?????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?쁬?@7?i?I?????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?쁬?@7?iF??`n????Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?쁬?@7?i??xV????Unknown
v%HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?쁬?@7?i¡??>????Unknown
?&HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?쁬?@7?i 2p?&????Unknown
?'HostMatMul",gradient_tape/sequential_68/dense_211/MatMul(1      @9      @A      @I      @a?쁬?@7?i>?E?????Unknown
?(HostBiasAddGrad"9gradient_tape/sequential_68/dense_212/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?쁬?@7?i|R??????Unknown
?)HostMatMul".gradient_tape/sequential_68/dense_213/MatMul_1(1      @9      @A      @I      @a?쁬?@7?i????ެ???Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?#?#2?2?iV562????Unknown
V+HostMean"Mean(1      @9      @A      @I      @a?#?#2?2?i??y|?????Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?#?#2?2?i?<??س???Unknown
b-HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?#?#2?2?iJ?	,????Unknown
?.HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?#?#2?2?i?#GO????Unknown
?/HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?#?#2?2?i???Һ???Unknown
?0Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?#?#2?2?iv
??%????Unknown
?1HostReadVariableOp".sequential_68/dense_212/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?#?#2?2?i?}"y????Unknown
v2Host_FusedMatMul"sequential_68/dense_212/Relu(1      @9      @A      @I      @a?#?#2?2?i>?Xh?????Unknown
y3Host_FusedMatMul"sequential_68/dense_213/BiasAdd(1      @9      @A      @I      @a?#?#2?2?i?d??????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??h5K?+?i-?P#?????Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a??h5K?+?i???????Unknown
?6HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      >@9      >@A      @I      @a??h5K?+?iCh?[????Unknown
?7HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??h5K?+?iξj?????Unknown
r8HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a??h5K?+?iY??????Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??h5K?+?i?k?j?????Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??h5K?+?io?T????Unknown
w;HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a??h5K?+?i?8T????Unknown
?<HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a??h5K?+?i?o???????Unknown
?=Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??h5K?+?iƞ=?????Unknown
?>HostBiasAddGrad"9gradient_tape/sequential_68/dense_211/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??h5K?+?i?R?N????Unknown
??HostReadVariableOp"-sequential_68/dense_211/MatMul/ReadVariableOp(1      @9      @A      @I      @a??h5K?+?i&s'????Unknown
?@HostReadVariableOp".sequential_68/dense_213/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??h5K?+?i?ɸ??????Unknown
tAHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?#?#2?"?ic?>?????Unknown
VBHostCast"Cast(1       @9       @A       @I       @a?#?#2?"?i=??????Unknown
XCHostCast"Cast_3(1       @9       @A       @I       @a?#?#2?"?i?v?H????Unknown
XDHostCast"Cast_5(1       @9       @A       @I       @a?#?#2?"?iy?A(r????Unknown
XEHostEqual"Equal(1       @9       @A       @I       @a?#?#2?"?i+?c˛????Unknown
sFHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?#?#2?"?i?#?n?????Unknown
jGHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?#?#2?"?i?]??????Unknown
vHHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?#?#2?"?iA?ʴ????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?#?#2?"?i???WB????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?#?#2?"?i?
?k????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?#?#2?"?iWD1??????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?#?#2?"?i	~SA?????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?#?#2?"?i??u??????Unknown
~NHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?#?#2?"?im???????Unknown
?OHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?#?#2?"?i+?*<????Unknown
?PHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?#?#2?"?i?d??e????Unknown
?QHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?#?#2?"?i???p?????Unknown
?RHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?#?#2?"?i5? ?????Unknown
~SHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?#?#2?"?i?C??????Unknown
?THostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?#?#2?"?i?KeZ????Unknown
?UHostReluGrad".gradient_tape/sequential_68/dense_211/ReluGrad(1       @9       @A       @I       @a?#?#2?"?iK???5????Unknown
?VHostReluGrad".gradient_tape/sequential_68/dense_212/ReluGrad(1       @9       @A       @I       @a?#?#2?"?i????_????Unknown
?WHostReadVariableOp"-sequential_68/dense_212/MatMul/ReadVariableOp(1       @9       @A       @I       @a?#?#2?"?i???C?????Unknown
?XHostReadVariableOp"-sequential_68/dense_213/MatMul/ReadVariableOp(1       @9       @A       @I       @a?#?#2?"?ia2???????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?#?#2??i:O?G????Unknown
vZHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?#?#2??il??????Unknown
X[HostCast"Cast_4(1      ??9      ??A      ??I      ??a?#?#2??i숡[q????Unknown
a\HostIdentity"Identity(1      ??9      ??A      ??I      ??a?#?#2??iť2-????Unknown?
T]HostMul"Mul(1      ??9      ??A      ??I      ??a?#?#2??i?????????Unknown
u^HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?#?#2??iw?T?/????Unknown
|_HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?#?#2??iP????????Unknown
?`HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?#?#2??i)wsY????Unknown
waHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?#?#2??i6E?????Unknown
ybHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?#?#2??i?R??????Unknown
?cHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?#?#2??i?o*?????Unknown
?dHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?#?#2??i?????????Unknown
?eHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?#?#2??if?L?A????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?#?#2??i???\?????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?#?#2??i?n.k????Unknown
?hHostReadVariableOp".sequential_68/dense_211/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?#?#2??i?????????Unknown
WiHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
YjHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?g
uHostFlushSummaryWriter"FlushSummaryWriter(1     H?@9     H?@A     H?@I     H?@a??B???i??B????Unknown?
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      E@9      E@A      E@I      E@a??~??^??i??Jc7????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      E@9      E@A     ?B@I     ?B@a?d?>????i?A-'r???Unknown
iHostWriteSummary"WriteSummary(1     ?B@9     ?B@A     ?B@I     ?B@a?d?>????i68?:???Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      ;@9      ;@A      4@I      4@aտlm???i5???)????Unknown
dHostAddN"SGD/gradients/AddN(1      3@9      3@A      3@I      3@a
?ZN䪉?iA-'r????Unknown
vHost_FusedMatMul"sequential_68/dense_211/Relu(1      3@9      3@A      3@I      3@a
?ZN䪉?iM?`?s???Unknown
gHostStridedSlice"strided_slice(1      3@9      3@A      3@I      3@a
?ZN䪉?iY??,????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      1@9      1@A      1@I      1@au	68???i??t	6???Unknown
t
HostSigmoid"sequential_68/dense_213/Sigmoid(1      *@9      *@A      *@I      *@aJ??ߏ??i؍*?H|???Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@aj9??f?}?iK????????Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      &@9      &@A      &@I      &@aj9??f?}?i??؍*????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aտlm?{?i>??4)???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aտlm?{?i???w=_???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a?FH/Qx?iJ??ߏ???Unknown?
?HostMatMul",gradient_tape/sequential_68/dense_212/MatMul(1      "@9      "@A      "@I      "@a?FH/Qx?i֦K??????Unknown
?HostMatMul".gradient_tape/sequential_68/dense_212/MatMul_1(1      "@9      "@A      "@I      "@a?FH/Qx?ib7??#????Unknown
?HostMatMul",gradient_tape/sequential_68/dense_213/MatMul(1      "@9      "@A      "@I      "@a?FH/Qx?i????!???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A       @I       @a??#?a?u?i??? M???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      T@9      T@A       @I       @a??#?a?u?i W?p;x???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a??#?a?u?i???4v????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??#?a?u?iR????????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aS????r?i???c?????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aS????r?i??]?W???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aS????r?iD??:+@???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aS????r?i??)??e???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a???t	6p?i???j????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a???t	6p?iPL??֦???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a???t	6p?i??B????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a???t	6p?i?????????Unknown
?HostBiasAddGrad"9gradient_tape/sequential_68/dense_213/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???t	6p?iim????Unknown
? HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aտlm?k?i)?'?#???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aտlm?k?i?F?y$>???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aտlm?k?i??4)Y???Unknown
~#HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aտlm?k?ii p?-t???Unknown
v$HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aտlm?k?i)?ݨ2????Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aտlm?k?i??Jc7????Unknown
?&HostMatMul",gradient_tape/sequential_68/dense_211/MatMul(1      @9      @A      @I      @aտlm?k?i?f?<????Unknown
?'HostBiasAddGrad"9gradient_tape/sequential_68/dense_212/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aտlm?k?ii?%?@????Unknown
?(HostMatMul".gradient_tape/sequential_68/dense_213/MatMul_1(1      @9      @A      @I      @aտlm?k?i)@??E????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??#?a?e?i?c??????Unknown
V*HostMean"Mean(1      @9      @A      @I      @a??#?a?e?iÇuV?&???Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??#?a?e?i??f?<???Unknown
b,HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??#?a?e?i]?W?Q???Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??#?a?e?i*?H|Xg???Unknown
?.HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??#?a?e?i?:??|???Unknown
?/Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a??#?a?e?i?:+@?????Unknown
?0HostReadVariableOp".sequential_68/dense_212/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??#?a?e?i?^?0????Unknown
v1Host_FusedMatMul"sequential_68/dense_212/Relu(1      @9      @A      @I      @a??#?a?e?i^?ν???Unknown
y2Host_FusedMatMul"sequential_68/dense_213/BiasAdd(1      @9      @A      @I      @a??#?a?e?i+??ek????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???t	6`?i?so?????Unknown
\4HostGreater"Greater(1      @9      @A      @I      @a???t	6`?i?[?x?????Unknown
?5HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      >@9      >@A      @I      @a???t	6`?i?6]????Unknown
?6HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???t	6`?i?ҋC???Unknown
r7HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a???t	6`?il?F?y$???Unknown
z8HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???t	6`?iFǻ??4???Unknown
v9HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a???t	6`?i ?0??D???Unknown
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a???t	6`?i?|??U???Unknown
?;HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a???t	6`?i?W?Qe???Unknown
?<Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a???t	6`?i?2?ću???Unknown
?=HostBiasAddGrad"9gradient_tape/sequential_68/dense_211/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???t	6`?i?ν????Unknown
?>HostReadVariableOp"-sequential_68/dense_211/MatMul/ReadVariableOp(1      @9      @A      @I      @a???t	6`?ib?x??????Unknown
??HostReadVariableOp".sequential_68/dense_213/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???t	6`?i<???)????Unknown
t@HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??#?a?U?i"U???????Unknown
VAHostCast"Cast(1       @9       @A       @I       @a??#?a?U?i??Bǻ???Unknown
XBHostCast"Cast_3(1       @9       @A       @I       @a??#?a?U?i?x???????Unknown
XCHostCast"Cast_5(1       @9       @A       @I       @a??#?a?U?i?
Фd????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @a??#?a?U?i???U3????Unknown
sEHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i?.?????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??#?a?U?i?????????Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??#?a?U?ilR?h?????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??#?a?U?iR??n???Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??#?a?U?i8v??<???Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i?{???Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??#?a?U?i??,?'???Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??#?a?U?i?+?ݨ2???Unknown
~MHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a??#?a?U?iн??w=???Unknown
?NHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??#?a?U?i?O~?FH???Unknown
?OHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a??#?a?U?i??v?S???Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??#?a?U?i?so??]???Unknown
?QHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??#?a?U?ihhR?h???Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??#?a?U?iN?`?s???Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??#?a?U?i4)Y?O~???Unknown
?THostReluGrad".gradient_tape/sequential_68/dense_211/ReluGrad(1       @9       @A       @I       @a??#?a?U?i?Qe????Unknown
?UHostReluGrad".gradient_tape/sequential_68/dense_212/ReluGrad(1       @9       @A       @I       @a??#?a?U?i MJ?????Unknown
?VHostReadVariableOp"-sequential_68/dense_212/MatMul/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i??Bǻ????Unknown
?WHostReadVariableOp"-sequential_68/dense_213/MatMul/ReadVariableOp(1       @9       @A       @I       @a??#?a?U?i?p;x?????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??#?a?E?i?????????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??#?a?E?i?4)Y????Unknown
XZHostCast"Cast_4(1      ??9      ??A      ??I      ??a??#?a?E?i?K???????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a??#?a?E?i??,?'????Unknown?
T\HostMul"Mul(1      ??9      ??A      ??I      ??a??#?a?E?i?ݨ2?????Unknown
u]HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??#?a?E?i~&%??????Unknown
|^HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a??#?a?E?iqo??]????Unknown
?_HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??#?a?E?id?<?????Unknown
w`HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??#?a?E?iW??,????Unknown
yaHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??#?a?E?iJJ??????Unknown
?bHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??#?a?E?i=??E?????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??#?a?E?i0??b????Unknown
?dHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??#?a?E?i#%???????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??#?a?E?inO1????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??#?a?E?i	????????Unknown
?gHostReadVariableOp".sequential_68/dense_211/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??#?a?E?i?????????Unknown
WhHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
YiHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU