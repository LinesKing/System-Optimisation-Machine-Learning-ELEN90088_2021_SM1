"??
BHostIDLE"IDLE1    ???@A    ???@a????`??i????`???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     T?@9     T?@A     T?@I     T?@aw?3@??i?1?^h???Unknown?
iHostWriteSummary"WriteSummary(1     ?B@9     ?B@A     ?B@I     ?B@a?3???i?i???l?????Unknown?
qHostSum" dense_283/kernel/Regularizer/Sum(1     ?@@9     ?@@A     ?@@I     ?@@a?l?ށg?i#a??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      5@9      5@A      5@I      5@a-???c]?i:<?̸????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      8@9      8@A      4@I      4@ak???v?[?i?*??????Unknown
vHost_FusedMatMul"sequential_91/dense_282/Relu(1      3@9      3@A      3@I      3@a?eɲ0?Z?i?@? ????Unknown
vHost_FusedMatMul"sequential_91/dense_283/Relu(1      .@9      .@A      .@I      .@a????T?i?? -?????Unknown
V	HostSum"Sum_2(1      *@9      *@A      *@I      *@a??1R?iF????????Unknown
?
HostSign"3gradient_tape/dense_284/kernel/Regularizer/Abs/Sign(1      (@9      (@A      (@I      (@a??-G?P?i??%? ????Unknown
dHostDataset"Iterator::Model(1      @@9      @@A      &@I      &@aÐ,~?N?i???????Unknown
qHostAbs" dense_284/kernel/Regularizer/Abs(1      &@9      &@A      &@I      &@aÐ,~?N?i??d?e????Unknown
?HostMul"0gradient_tape/dense_284/kernel/Regularizer/Mul_1(1      $@9      $@A      $@I      $@ak???v?K?i???d????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      ,@9      ,@A      "@I      "@aS??0I?i???0?????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?:f?^dF?i&?wHJ???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?:f?^dF?i??1`????Unknown?
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?:f?^dF?iD??w|???Unknown
?HostMatMul",gradient_tape/sequential_91/dense_284/MatMul(1       @9       @A       @I       @a?:f?^dF?i?a?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aesy
ӗC?i0 h?????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aesy
ӗC?i??*y????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aesy
ӗC?i?<?m? ???Unknown
?HostMatMul",gradient_tape/sequential_91/dense_283/MatMul(1      @9      @A      @I      @aesy
ӗC?iGۯb?%???Unknown
yHost_FusedMatMul"sequential_91/dense_284/BiasAdd(1      @9      @A      @I      @aesy
ӗC?i?yrW?*???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aesy
ӗC?i5Ly/???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??-G?@?i,{ ?3???Unknown
?HostSign"3gradient_tape/dense_283/kernel/Regularizer/Abs/Sign(1      @9      @A      @I      @a??-G?@?iW????7???Unknown
?HostMatMul",gradient_tape/sequential_91/dense_282/MatMul(1      @9      @A      @I      @a??-G?@?i?A??<???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_91/dense_283/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??-G?@?i??b?D@???Unknown
?HostMatMul".gradient_tape/sequential_91/dense_283/MatMul_1(1      @9      @A      @I      @a??-G?@?i?.ewD???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      G@9      G@A      @I      @ak???v?;?i?/?G???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ak???v?;?i?W??vK???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ak???v?;?i??q?N???Unknown
s!HostSum""dense_282/kernel/Regularizer/Sum_1(1      @9      @A      @I      @ak???v?;?i??~ vR???Unknown
s"HostSum""dense_283/kernel/Regularizer/Sum_1(1      @9      @A      @I      @ak???v?;?i??R??U???Unknown
s#HostSum""dense_284/kernel/Regularizer/Sum_1(1      @9      @A      @I      @ak???v?;?i??&~uY???Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @ak???v?;?i??,?\???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?:f?^d6?inظ?_???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?:f?^d6?i5??D?b???Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?:f?^d6?i????Ze???Unknown
s(HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?:f?^d6?i??n\'h???Unknown
?)HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?:f?^d6?i??K??j???Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?:f?^d6?iQ?(t?m???Unknown
q+HostSum" dense_284/kernel/Regularizer/Sum(1      @9      @A      @I      @a?:f?^d6?i? ?p???Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?:f?^d6?i߅??Ys???Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?:f?^d6?i?r?&v???Unknown
?.HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?:f?^d6?im_???x???Unknown
?/HostMul"0gradient_tape/dense_282/kernel/Regularizer/Mul_1(1      @9      @A      @I      @a?:f?^d6?i4Ly/?{???Unknown
?0HostMul"0gradient_tape/dense_283/kernel/Regularizer/Mul_1(1      @9      @A      @I      @a?:f?^d6?i?8V??~???Unknown
?1HostBiasAddGrad"9gradient_tape/sequential_91/dense_282/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?:f?^d6?i?%3GX????Unknown
?2HostReluGrad".gradient_tape/sequential_91/dense_283/ReluGrad(1      @9      @A      @I      @a?:f?^d6?i??$????Unknown
?3HostBiasAddGrad"9gradient_tape/sequential_91/dense_284/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?:f?^d6?iP??^?????Unknown
?4HostReadVariableOp".sequential_91/dense_282/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?:f?^d6?i??꽉???Unknown
t5HostSigmoid"sequential_91/dense_284/Sigmoid(1      @9      @A      @I      @a?:f?^d6?i?ئv?????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??-G?0?it??ߣ????Unknown
\7HostGreater"Greater(1      @9      @A      @I      @a??-G?0?i
<rH?????Unknown
?8HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      @I      @a??-G?0?i??W?֒???Unknown
V9HostMean"Mean(1      @9      @A      @I      @a??-G?0?i6?=?????Unknown
u:HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??-G?0?i?P#?	????Unknown
?;HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??-G?0?ib	?"????Unknown
?<HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??-G?0?i???T<????Unknown
?=HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??-G?0?i?eԽU????Unknown
z>HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??-G?0?i$?&o????Unknown
v?HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??-G?0?i?ȟ??????Unknown
|@HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??-G?0?iPz???????Unknown
vAHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??-G?0?i?+ka?????Unknown
vBHostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??-G?0?i|?P?ԧ???Unknown
vCHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??-G?0?i?63?????Unknown
qDHostAbs" dense_282/kernel/Regularizer/Abs(1      @9      @A      @I      @a??-G?0?i?@?????Unknown
qEHostSum" dense_282/kernel/Regularizer/Sum(1      @9      @A      @I      @a??-G?0?i>?!????Unknown
qFHostAbs" dense_283/kernel/Regularizer/Abs(1      @9      @A      @I      @a??-G?0?iԣ?m:????Unknown
wGHostSquare"#dense_283/kernel/Regularizer/Square(1      @9      @A      @I      @a??-G?0?ijU??S????Unknown
?HHostReadVariableOp"/dense_284/kernel/Regularizer/Abs/ReadVariableOp(1      @9      @A      @I      @a??-G?0?i ??m????Unknown
wIHostSquare"#dense_284/kernel/Regularizer/Square(1      @9      @A      @I      @a??-G?0?i?????????Unknown
bJHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??-G?0?i,j~?????Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??-G?0?i?dz?????Unknown
?LHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??-G?0?iX?I?Ҽ???Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a??-G?0?i?~/L?????Unknown
?NHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??-G?0?i?0?????Unknown
?OHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a??-G?0?i??????Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??-G?0?i????8????Unknown
?QHostMatMul".gradient_tape/sequential_91/dense_284/MatMul_1(1      @9      @A      @I      @a??-G?0?iFE??Q????Unknown
?RHostReadVariableOp".sequential_91/dense_283/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??-G?0?i???Xk????Unknown
zSHostAddN"(ArithmeticOptimizer/AddOpsRewrite_AddN_1(1       @9       @A       @I       @a?:f?^d&?i@m???????Unknown
tTHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?:f?^d&?i????7????Unknown
VUHostCast"Cast(1       @9       @A       @I       @a?:f?^d&?iZw*?????Unknown
XVHostCast"Cast_3(1       @9       @A       @I       @a?:f?^d&?il?ep????Unknown
XWHostCast"Cast_5(1       @9       @A       @I       @a?:f?^d&?i?FT?j????Unknown
XXHostEqual"Equal(1       @9       @A       @I       @a?:f?^d&?i4?B??????Unknown
aYHostIdentity"Identity(1       @9       @A       @I       @a?:f?^d&?i?31B7????Unknown?
?ZHostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a?:f?^d&?i????????Unknown
d[HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?:f?^d&?i` ?????Unknown
f\HostAddN"SGD/gradients/AddN_1(1       @9       @A       @I       @a?:f?^d&?iĖ?j????Unknown
f]HostAddN"SGD/gradients/AddN_3(1       @9       @A       @I       @a?:f?^d&?i(?Y?????Unknown
j^HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?:f?^d&?i??ٟ6????Unknown
r_HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?:f?^d&?i?????????Unknown
v`HostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?:f?^d&?iTp?+????Unknown
?aHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?:f?^d&?i???qi????Unknown
wbHostSquare"#dense_282/kernel/Regularizer/Square(1       @9       @A       @I       @a?:f?^d&?i]???????Unknown
?cHostReadVariableOp"2dense_283/kernel/Regularizer/Square/ReadVariableOp(1       @9       @A       @I       @a?:f?^d&?i?Ӂ?5????Unknown
?dHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?:f?^d&?i?IpC?????Unknown
xeHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?:f?^d&?iH?^?????Unknown
?fHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?:f?^d&?i?6M?h????Unknown
?gHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?:f?^d&?i?;?????Unknown
~hHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?:f?^d&?it#*[5????Unknown
?iHostSign"3gradient_tape/dense_282/kernel/Regularizer/Abs/Sign(1       @9       @A       @I       @a?:f?^d&?iؙ??????Unknown
?jHostReluGrad".gradient_tape/sequential_91/dense_282/ReluGrad(1       @9       @A       @I       @a?:f?^d&?i<?????Unknown
vkHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?:f?^d?inK?	?????Unknown
XlHostCast"Cast_4(1      ??9      ??A      ??I      ??a?:f?^d?i???,h????Unknown
TmHostMul"Mul(1      ??9      ??A      ??I      ??a?:f?^d?i???O????Unknown
|nHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i??r?????Unknown
}oHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?:f?^d?i68ە?????Unknown
?pHostReadVariableOp"/dense_282/kernel/Regularizer/Abs/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?ihsҸ4????Unknown
qqHostMul" dense_282/kernel/Regularizer/mul(1      ??9      ??A      ??I      ??a?:f?^d?i?????????Unknown
srHostMul""dense_282/kernel/Regularizer/mul_1(1      ??9      ??A      ??I      ??a?:f?^d?i?????????Unknown
?sHostReadVariableOp"/dense_283/kernel/Regularizer/Abs/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i?$?!N????Unknown
qtHostMul" dense_283/kernel/Regularizer/mul(1      ??9      ??A      ??I      ??a?:f?^d?i0`?D????Unknown
suHostMul""dense_283/kernel/Regularizer/mul_1(1      ??9      ??A      ??I      ??a?:f?^d?ib??g?????Unknown
?vHostReadVariableOp"2dense_284/kernel/Regularizer/Square/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i?֝?g????Unknown
qwHostMul" dense_284/kernel/Regularizer/mul(1      ??9      ??A      ??I      ??a?:f?^d?i???????Unknown
sxHostMul""dense_284/kernel/Regularizer/mul_1(1      ??9      ??A      ??I      ??a?:f?^d?i?L???????Unknown
`yHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?:f?^d?i*????????Unknown
wzHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i\?z4????Unknown
?{HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?:f?^d?i??q9?????Unknown
?|Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?:f?^d?i?9i\?????Unknown
?}HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?:f?^d?i?t`M????Unknown
?~HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?:f?^d?i$?W? ????Unknown
?HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?:f?^d?iV?Nų????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?:f?^d?i?&F?f????Unknown
??HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?:f?^d?i?a=????Unknown
??HostMul"2gradient_tape/dense_282/kernel/Regularizer/Abs/mul(1      ??9      ??A      ??I      ??a?:f?^d?i??4.?????Unknown
??HostMul"2gradient_tape/dense_283/kernel/Regularizer/Abs/mul(1      ??9      ??A      ??I      ??a?:f?^d?i?+Q?????Unknown
??HostMul"2gradient_tape/dense_284/kernel/Regularizer/Abs/mul(1      ??9      ??A      ??I      ??a?:f?^d?iP#t3????Unknown
??HostReadVariableOp"-sequential_91/dense_282/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i?N??????Unknown
??HostReadVariableOp"-sequential_91/dense_283/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i????????Unknown
??HostReadVariableOp".sequential_91/dense_284/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i???L????Unknown
??HostReadVariableOp"-sequential_91/dense_284/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?:f?^d?i     ???Unknown
:?HostAddN"SGD/gradients/AddN_2(i     ???Unknown
b?HostReadVariableOp"2dense_282/kernel/Regularizer/Square/ReadVariableOp(i     ???Unknown
I?HostReadVariableOp"div_no_nan/ReadVariableOp(i     ???Unknown
K?HostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown
M?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
\?HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
Z?HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown*??
uHostFlushSummaryWriter"FlushSummaryWriter(1     T?@9     T?@A     T?@I     T?@aI!?W???iI!?W????Unknown?
iHostWriteSummary"WriteSummary(1     ?B@9     ?B@A     ?B@I     ?B@a?	??vd??i??v{????Unknown?
qHostSum" dense_283/kernel/Regularizer/Sum(1     ?@@9     ?@@A     ?@@I     ?@@aP$?Ҽ???i???\AL???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      5@9      5@A      5@I      5@af???k??iZ???????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      8@9      8@A      4@I      4@a????A5??i1???????Unknown
vHost_FusedMatMul"sequential_91/dense_282/Relu(1      3@9      3@A      3@I      3@a?e???i=?2t?n???Unknown
vHost_FusedMatMul"sequential_91/dense_283/Relu(1      .@9      .@A      .@I      .@aI8?y?'??ik:`????Unknown
VHostSum"Sum_2(1      *@9      *@A      *@I      *@aӥb[ox?ij0?Q????Unknown
?	HostSign"3gradient_tape/dense_284/kernel/Regularizer/Abs/Sign(1      (@9      (@A      (@I      (@at?n??}?i???j0???Unknown
d
HostDataset"Iterator::Model(1      @@9      @@A      &@I      &@a?z???z?i?v{?e???Unknown
qHostAbs" dense_284/kernel/Regularizer/Abs(1      &@9      &@A      &@I      &@a?z???z?iW??r?????Unknown
?HostMul"0gradient_tape/dense_284/kernel/Regularizer/Mul_1(1      $@9      $@A      $@I      $@a????A5x?iB??X????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      ,@9      ,@A      "@I      "@aW?+??u?ic-C?????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?*?_?]s?i?k?????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?*?_?]s?i??@cD???Unknown?
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?*?_?]s?ie???k???Unknown
?HostMatMul",gradient_tape/sequential_91/dense_284/MatMul(1       @9       @A       @I       @a?*?_?]s?i?&@zڑ???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?E???p?iF}g??????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?E???p?i?ӎ̢????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?E???p?i\*???????Unknown
?HostMatMul",gradient_tape/sequential_91/dense_283/MatMul(1      @9      @A      @I      @a?E???p?i???k???Unknown
yHost_FusedMatMul"sequential_91/dense_284/BiasAdd(1      @9      @A      @I      @a?E???p?ir?HO;???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a?E???p?i?-,q3]???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??m?i???&@z???Unknown
?HostSign"3gradient_tape/dense_283/kernel/Regularizer/Abs/Sign(1      @9      @A      @I      @at?n??m?i}K?L????Unknown
?HostMatMul",gradient_tape/sequential_91/dense_282/MatMul(1      @9      @A      @I      @at?n??m?i=zڑY????Unknown
?HostBiasAddGrad"9gradient_tape/sequential_91/dense_283/BiasAdd/BiasAddGrad(1      @9      @A      @I      @at?n??m?i??iGf????Unknown
?HostMatMul".gradient_tape/sequential_91/dense_283/MatMul_1(1      @9      @A      @I      @at?n??m?i?W??r????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      G@9      G@A      @I      @a????A5h?i???>????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a????A5h?i?e??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a????A5h?i????7???Unknown
s HostSum""dense_282/kernel/Regularizer/Sum_1(1      @9      @A      @I      @a????A5h?i?s?HO???Unknown
s!HostSum""dense_283/kernel/Regularizer/Sum_1(1      @9      @A      @I      @a????A5h?i???F}g???Unknown
s"HostSum""dense_284/kernel/Regularizer/Sum_1(1      @9      @A      @I      @a????A5h?i??ƈ????Unknown
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a????A5h?iw???????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?*?_?]c?i???E????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?*?_?]c?i?F}g?????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?*?_?]c?i???5????Unknown
s'HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?*?_?]c?i#?<_????Unknown
?(HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?*?_?]c?iN$?Ҽ????Unknown
~)HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?*?_?]c?iy??????Unknown
q*HostSum" dense_284/kernel/Regularizer/Sum(1      @9      @A      @I      @a?*?_?]c?i?b[ox???Unknown
?+HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?*?_?]c?i??=?2???Unknown
?,HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?*?_?]c?i??4F???Unknown
?-HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?*?_?]c?i%@zڑY???Unknown
?.HostMul"0gradient_tape/dense_282/kernel/Regularizer/Mul_1(1      @9      @A      @I      @a?*?_?]c?iP?٨?l???Unknown
?/HostMul"0gradient_tape/dense_283/kernel/Regularizer/Mul_1(1      @9      @A      @I      @a?*?_?]c?i{~9wM????Unknown
?0HostBiasAddGrad"9gradient_tape/sequential_91/dense_282/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?*?_?]c?i??E?????Unknown
?1HostReluGrad".gradient_tape/sequential_91/dense_283/ReluGrad(1      @9      @A      @I      @a?*?_?]c?iѼ?	????Unknown
?2HostBiasAddGrad"9gradient_tape/sequential_91/dense_284/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?*?_?]c?i?[X?f????Unknown
?3HostReadVariableOp".sequential_91/dense_282/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?*?_?]c?i'????????Unknown
t4HostSigmoid"sequential_91/dense_284/Sigmoid(1      @9      @A      @I      @a?*?_?]c?iR?"????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @at?n??]?i?Q?٨????Unknown
\6HostGreater"Greater(1      @9      @A      @I      @at?n??]?i	?4/????Unknown
?7HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      1@9      1@A      @I      @at?n??]?ir?n?????Unknown
V8HostMean"Mean(1      @9      @A      @I      @at?n??]?i?w6?;???Unknown
u9HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @at?n??]?i2/?D?)???Unknown
?:HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??]?i??şH8???Unknown
?;HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??]?i?????F???Unknown
?<HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @at?n??]?iRUUUUU???Unknown
z=HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @at?n??]?i???c???Unknown
v>HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @at?n??]?i??
br???Unknown
|?HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @at?n??]?ir{?e?????Unknown
v@HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @at?n??]?i?2t?n????Unknown
vAHostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @at?n??]?i2?;?????Unknown
vBHostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @at?n??]?i??v{????Unknown
qCHostAbs" dense_282/kernel/Regularizer/Abs(1      @9      @A      @I      @at?n??]?i?X??????Unknown
qDHostSum" dense_282/kernel/Regularizer/Sum(1      @9      @A      @I      @at?n??]?iR?+?????Unknown
qEHostAbs" dense_283/kernel/Regularizer/Abs(1      @9      @A      @I      @at?n??]?i??Z?????Unknown
wFHostSquare"#dense_283/kernel/Regularizer/Square(1      @9      @A      @I      @at?n??]?i"??????Unknown
?GHostReadVariableOp"/dense_284/kernel/Regularizer/Abs/ReadVariableOp(1      @9      @A      @I      @at?n??]?ir6?;????Unknown
wHHostSquare"#dense_284/kernel/Regularizer/Square(1      @9      @A      @I      @at?n??]?i????????Unknown
bIHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @at?n??]?i2?y?'???Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @at?n??]?i?\AL? ???Unknown
?KHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @at?n??]?i?	?4/???Unknown
?LHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @at?n??]?iR???=???Unknown
?MHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @at?n??]?i???\AL???Unknown
?NHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @at?n??]?i:`??Z???Unknown
?OHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @at?n??]?ir?'Ni???Unknown
?PHostMatMul".gradient_tape/sequential_91/dense_284/MatMul_1(1      @9      @A      @I      @at?n??]?iҨ?l?w???Unknown
?QHostReadVariableOp".sequential_91/dense_283/BiasAdd/ReadVariableOp(1      @9      @A      @I      @at?n??]?i2`??Z????Unknown
zRHostAddN"(ArithmeticOptimizer/AddOpsRewrite_AddN_1(1       @9       @A       @I       @a?*?_?]S?i?/??	????Unknown
tSHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?*?_?]S?i\???????Unknown
VTHostCast"Cast(1       @9       @A       @I       @a?*?_?]S?i??F}g????Unknown
XUHostCast"Cast_3(1       @9       @A       @I       @a?*?_?]S?i??vd????Unknown
XVHostCast"Cast_5(1       @9       @A       @I       @a?*?_?]S?in?KŶ???Unknown
XWHostEqual"Equal(1       @9       @A       @I       @a?*?_?]S?i?=?2t????Unknown
aXHostIdentity"Identity(1       @9       @A       @I       @a?*?_?]S?iE#????Unknown?
?YHostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a?*?_?]S?i??5?????Unknown
dZHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?*?_?]S?io?e??????Unknown
f[HostAddN"SGD/gradients/AddN_1(1       @9       @A       @I       @a?*?_?]S?i|??/????Unknown
f\HostAddN"SGD/gradients/AddN_3(1       @9       @A       @I       @a?*?_?]S?i?KŶ?????Unknown
j]HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?*?_?]S?i.???????Unknown
r^HostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?*?_?]S?i??$?<???Unknown
v_HostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?*?_?]S?iX?Tl????Unknown
?`HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?*?_?]S?i퉄S????Unknown
waHostSquare"#dense_282/kernel/Regularizer/Square(1       @9       @A       @I       @a?*?_?]S?i?Y?:I!???Unknown
?bHostReadVariableOp"2dense_283/kernel/Regularizer/Square/ReadVariableOp(1       @9       @A       @I       @a?*?_?]S?i)?!?*???Unknown
?cHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?*?_?]S?i??	?4???Unknown
xdHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?*?_?]S?iA?C?U>???Unknown
?eHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?*?_?]S?i֗s?H???Unknown
?fHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?*?_?]S?ikg???Q???Unknown
~gHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?*?_?]S?i 7ӥb[???Unknown
?hHostSign"3gradient_tape/dense_282/kernel/Regularizer/Abs/Sign(1       @9       @A       @I       @a?*?_?]S?i??e???Unknown
?iHostReluGrad".gradient_tape/sequential_91/dense_282/ReluGrad(1       @9       @A       @I       @a?*?_?]S?i*?2t?n???Unknown
vjHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?*?_?]C?i?????s???Unknown
XkHostCast"Cast_4(1      ??9      ??A      ??I      ??a?*?_?]C?i??b[ox???Unknown
TlHostMul"Mul(1      ??9      ??A      ??I      ??a?*?_?]C?i????F}???Unknown
|mHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?iVu?B????Unknown
}nHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?*?_?]C?i!]*??????Unknown
?oHostReadVariableOp"/dense_282/kernel/Regularizer/Abs/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i?D?)͋???Unknown
qpHostMul" dense_282/kernel/Regularizer/mul(1      ??9      ??A      ??I      ??a?*?_?]C?i?,Z??????Unknown
sqHostMul""dense_282/kernel/Regularizer/mul_1(1      ??9      ??A      ??I      ??a?*?_?]C?i??|????Unknown
?rHostReadVariableOp"/dense_283/kernel/Regularizer/Abs/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?iM???S????Unknown
qsHostMul" dense_283/kernel/Regularizer/mul(1      ??9      ??A      ??I      ??a?*?_?]C?i?!?*????Unknown
stHostMul""dense_283/kernel/Regularizer/mul_1(1      ??9      ??A      ??I      ??a?*?_?]C?i?˹k????Unknown
?uHostReadVariableOp"2dense_284/kernel/Regularizer/Square/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i??Q?٨???Unknown
qvHostMul" dense_284/kernel/Regularizer/mul(1      ??9      ??A      ??I      ??a?*?_?]C?iy??R?????Unknown
swHostMul""dense_284/kernel/Regularizer/mul_1(1      ??9      ??A      ??I      ??a?*?_?]C?iD??ƈ????Unknown
`xHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?*?_?]C?ik:`????Unknown
wyHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i?R??7????Unknown
?zHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?*?_?]C?i?:I!????Unknown
?{Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?*?_?]C?ip"???????Unknown
?|HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?*?_?]C?i;
y?????Unknown
?}HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?*?_?]C?i?|?????Unknown
?~HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?*?_?]C?i?٨?l????Unknown
?HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?*?_?]C?i??@cD????Unknown
??HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?*?_?]C?ig???????Unknown
??HostMul"2gradient_tape/dense_282/kernel/Regularizer/Abs/mul(1      ??9      ??A      ??I      ??a?*?_?]C?i2?pJ?????Unknown
??HostMul"2gradient_tape/dense_283/kernel/Regularizer/Abs/mul(1      ??9      ??A      ??I      ??a?*?_?]C?i?x??????Unknown
??HostMul"2gradient_tape/dense_284/kernel/Regularizer/Abs/mul(1      ??9      ??A      ??I      ??a?*?_?]C?i?`?1?????Unknown
??HostReadVariableOp"-sequential_91/dense_282/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i?H8?y????Unknown
??HostReadVariableOp"-sequential_91/dense_283/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i^0?Q????Unknown
??HostReadVariableOp".sequential_91/dense_284/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i)h?(????Unknown
??HostReadVariableOp"-sequential_91/dense_284/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?*?_?]C?i?????????Unknown
:?HostAddN"SGD/gradients/AddN_2(i?????????Unknown
b?HostReadVariableOp"2dense_282/kernel/Regularizer/Square/ReadVariableOp(i?????????Unknown
I?HostReadVariableOp"div_no_nan/ReadVariableOp(i?????????Unknown
K?HostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
M?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
\?HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
Z?HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU