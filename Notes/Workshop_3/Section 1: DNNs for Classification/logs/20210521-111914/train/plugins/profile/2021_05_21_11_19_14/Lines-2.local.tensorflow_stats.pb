"?o
BHostIDLE"IDLE1     a?@A     a?@av???z??iv???z???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      o@9      o@A     ?f@I     ?f@a?!ײS ??i"?&??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      `@9      `@A      `@I      `@a?:i;???i?1)?K???Unknown
?HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1      `@9      `@A     ?_@I     ?_@a/,j ???i[#S,?	???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ]@9      ]@A      ]@I      ]@a?Zk???i1.?W?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate(1      L@9      L@A      L@I      L@a$=kȵ???i&??.?
???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      H@9      H@A      H@I      H@a??>w???iA&?dR???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?E@9     ?E@A     ?E@I     ?E@a?|?'0??i4?c̨????Unknown?
s	HostReadVariableOp"SGD/Cast/ReadVariableOp(1     ?C@9     ?C@A     ?C@I     ?C@a߂??!%}?i:Q??????Unknown
h
HostRandomShuffle"RandomShuffle(1      8@9      8@A      8@I      8@a??>w?q?i?vi??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      6@9      6@A      6@I      6@aw????pp?i?C[?????Unknown
dHostDataset"Iterator::Model(1     `b@9     `b@A      2@I      2@a	*?2?j?i`8??,???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      2@9      2@A      2@I      2@a	*?2?j?i2|?G???Unknown
?HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      2@9      2@A      2@I      2@a	*?2?j?i\??Fib???Unknown
?HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      2@9      2@A      2@I      2@a	*?2?j?i???yP}???Unknown
^HostGatherV2"GatherV2(1      .@9      .@A      .@I      .@a]x?Ukf?i?K?λ????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      .@9      .@A      .@I      .@a]x?Ukf?iv??#'????Unknown
hHostTensorDataset"TensorDataset(1      ,@9      ,@A      ,@I      ,@a$=kȵ?d?i?N??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      *@9      *@A      *@I      *@a???nc?i??6??????Unknown
iHostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@a???nc?i?̹?????Unknown?
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      *@9      *@A      *@I      *@a???nc?i?=^????Unknown
XHostSlice"Slice(1      (@9      (@A      (@I      (@a??>w?a?i?{?M???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      (@9      (@A      (@I      (@a??>w?a?iG1?=???Unknown
`HostGatherV2"
GatherV2_1(1      &@9      &@A      &@I      &@aw????p`?i????-???Unknown
?HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1      &@9      &@A      &@I      &@aw????p`?i]???>???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      &@9      &@A      &@I      &@aw????p`?i?䣓?N???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      $@9      $@A      $@I      $@a|?tgq?]?i8?Ẃ]???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a|?tgq?]?i?Ytl???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a	*?2?Z?i??y??y???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      "@9      "@A      "@I      "@a	*?2?Z?i?u?7[????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      "@9      "@A      "@I      "@a	*?2?Z?i?W?Δ???Unknown
{ HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a	*?2?Z?iܑ?jB????Unknown
g!HostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a	*?2?Z?i?4?????Unknown
r"HostTensorSliceDataset"TensorSliceDataset(1       @9       @A       @I       @a???R??W?iˁ]??????Unknown
?#HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1       @9       @A       @I       @a???R??W?i?????????Unknown
t$Host_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @a???R??W?iE???????Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a$=kȵ?T?i{?M????Unknown
?&HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??>w?Q?i??3	????Unknown
|'HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??>w?Q?i?????????Unknown
?(HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??>w?Q?iG?q??????Unknown
?)HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??>w?Q?i??<????Unknown
r*HostConcatenateDataset"ConcatenateDataset(1      @9      @A      @I      @a|?tgq?M?i?}jXc	???Unknown
?+HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a|?tgq?M?i?Z?t????Unknown
?,HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a|?tgq?M?i"8?U???Unknown
V-HostSum"Sum_2(1      @9      @A      @I      @a|?tgq?M?iJx?????Unknown
z.HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a|?tgq?M?ir???G'???Unknown
v/HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a|?tgq?M?i??+??.???Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a???R??G?i??@c?4???Unknown
V1HostMean"Mean(1      @9      @A      @I      @a???R??G?it1U??:???Unknown
v2HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a???R??G?ia?i]?@???Unknown
v3HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a???R??G?iN?~ڪF???Unknown
u4HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a???R??G?i;D?W?L???Unknown
b5HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a???R??G?i(??ԟR???Unknown
?6HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???R??G?i??Q?X???Unknown
?7HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???R??G?iW?Δ^???Unknown
o8HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a???R??G?i??K?d???Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??>w?A?i???)i???Unknown
X:HostEqual"Equal(1      @9      @A      @I      @a??>w?A?iS??m???Unknown
\;HostGreater"Greater(1      @9      @A      @I      @a??>w?A?i?T?r???Unknown
e<Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a??>w?A?i?$?~v???Unknown?
Z=HostSlice"Slice_1(1      @9      @A      @I      @a??>w?A?ii????z???Unknown
?>HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??>w?A?i$?~v???Unknown
v?HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??>w?A?iͨ?\?????Unknown
v@HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??>w?A?i-b:n????Unknown
~AHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??>w?A?i1?1?????Unknown
?BHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??>w?A?i?6?e????Unknown
}CHostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a??>w?A?i?????????Unknown
?DHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??>w?A?iG@??]????Unknown
?EHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a??>w?A?i??o?ٞ???Unknown
VFHostCast"Cast(1       @9       @A       @I       @a???R??7?io??֡???Unknown
XGHostCast"Cast_3(1       @9       @A       @I       @a???R??7?i?u?Ԥ???Unknown
XHHostCast"Cast_5(1       @9       @A       @I       @a???R??7?i[?Kѧ???Unknown
uIHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a???R??7?i?&??Ϊ???Unknown
?JHostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a???R??7?iG#?˭???Unknown
dKHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a???R??7?i?׭ɰ???Unknown
jLHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a???R??7?i308EƳ???Unknown
rMHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a???R??7?i??ö???Unknown
}NHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a???R??7?i?L??????Unknown
`OHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a???R??7?i?9? ?????Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a???R??7?i?a??????Unknown
?QHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a???R??7?i???}?????Unknown
xRHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a???R??7?i?Bv??????Unknown
?SHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a???R??7?im? ??????Unknown
?THostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a???R??7?i???9?????Unknown
?UHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a???R??7?iYLx?????Unknown
?VHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a???R??7?iϤ???????Unknown
?WHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a???R??7?iE?)??????Unknown
?XHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a???R??7?i?U?3?????Unknown
~YHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a???R??7?i1?>r?????Unknown
}ZHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a???R??7?i?ɰ?????Unknown
[HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a???R??7?i_S??????Unknown
?\HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a???R??7?i???-?????Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a???R??'?i??"?????Unknown
v^HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a???R??'?i	hl?????Unknown
X_HostCast"Cast_4(1      ??9      ??A      ??I      ??a???R??'?iD<?????Unknown
a`HostIdentity"Identity(1      ??9      ??A      ??I      ??a???R??'?ih???????Unknown?
?aHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1      ??9      ??A      ??I      ??a???R??'?i??7J????Unknown
?bHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a???R??'?i??|??????Unknown
TcHostMul"Mul(1      ??9      ??A      ??I      ??a???R??'?i0???????Unknown
|dHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a???R??'?ik(?????Unknown
weHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???R??'?i?EL?????Unknown
yfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???R??'?i?q?f?????Unknown
?gHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a???R??'?i??????Unknown
?hHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a???R??'?iW???????Unknown
?iHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a???R??'?i??`D????Unknown
?jHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a???R??'?i?"???????Unknown
?kHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a???R??'?iO??????Unknown
?lHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a???R??'?iC{0"?????Unknown
?mHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a???R??'?i~?u?????Unknown
?nHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a???R??'?i?Ӻ`?????Unknown
?oHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a???R??'?i?????????Unknown
vpHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor(i?????????Unknown
iqHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
[rHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown
[sHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown*?n
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      o@9      o@A     ?f@I     ?f@a!Y?B??i!Y?B???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      `@9      `@A      `@I      `@a????ƣ??itW)&????Unknown
?HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1      `@9      `@A     ?_@I     ?_@a?.>??i??2??????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ]@9      ]@A      ]@I      ]@a(?ˀO??i??e?'????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate(1      L@9      L@A      L@I      L@ad!Y?B??i?'?ˀO???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      H@9      H@A      H@I      H@aV?	????i?,d!Y???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?E@9     ?E@A     ?E@I     ?E@a틨????iD?ξ?j???Unknown?
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1     ?C@9     ?C@A     ?C@I     ?C@àO???iJ1Aw?b???Unknown
h	HostRandomShuffle"RandomShuffle(1      8@9      8@A      8@I      8@aV?	????i?}?:????Unknown
{
HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      6@9      6@A      6@I      6@aϾ?j?}??i??e?'????Unknown
dHostDataset"Iterator::Model(1     `b@9     `b@A      2@I      2@a?O????i???ƣ????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      2@9      2@A      2@I      2@a?O????i6?l???Unknown
?HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      2@9      2@A      2@I      2@a?O????iMozӛ????Unknown
?HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      2@9      2@A      2@I      2@a?O????i????Q???Unknown
^HostGatherV2"GatherV2(1      .@9      .@A      .@I      .@a?싨?ه?i??x4????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      .@9      .@A      .@I      .@a?싨?ه?i??????Unknown
hHostTensorDataset"TensorDataset(1      ,@9      ,@A      ,@I      ,@ad!Y?B??iy4??h???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      *@9      *@A      *@I      *@a?U?	????i?]???????Unknown
iHostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@a?U?	????i'?ˀO???Unknown?
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      *@9      *@A      *@I      *@a?U?	????i~??h?`???Unknown
XHostSlice"Slice(1      (@9      (@A      (@I      (@aV?	????i???Q????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      (@9      (@A      (@I      (@aV?	????i???ƣ????Unknown
`HostGatherV2"
GatherV2_1(1      &@9      &@A      &@I      &@aϾ?j?}??i?l<?????Unknown
?HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1      &@9      &@A      &@I      &@aϾ?j?}??i?B??????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      &@9      &@A      &@I      &@aϾ?j?}??i?e?'?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      $@9      $@A      $@I      $@a??6??i??,d!???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a??6??i[????J???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?O??|?i??ƣ?????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      "@9      "@A      "@I      "@a?O??|?i????7????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      "@9      "@A      "@I      "@a?O??|?i8?"?u????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a?O??|?i?Q??/???Unknown
g HostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?O??|?iv4??h???Unknown
r!HostTensorSliceDataset"TensorSliceDataset(1       @9       @A       @I       @ar???py?i?Mozӛ???Unknown
?"HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1       @9       @A       @I       @ar???py?iXg_D?????Unknown
t#Host_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @ar???py?iɀO????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @ad!Y?Bv?i??.???Unknown
?%HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aV?	??s?i!?u?ET???Unknown
|&HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aV?	??s?i6??Moz???Unknown
?'HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aV?	??s?iK?]??????Unknown
?(HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aV?	??s?i`????????Unknown
r)HostConcatenateDataset"ConcatenateDataset(1      @9      @A      @I      @a??6?o?iG??????Unknown
?*HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??6?o?i.>9\???Unknown
?+HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??6?o?itW)&???Unknown
V,HostSum"Sum_2(1      @9      @A      @I      @a??6?o?i?"?u?E???Unknown
z-HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??6?o?i?2???e???Unknown
v.HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??6?o?i?B??????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @ar???pi?i?O?????Unknown
V0HostMean"Mean(1      @9      @A      @I      @ar???pi?i:\|r????Unknown
v1HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @ar???pi?i?h?`?????Unknown
v2HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @ar???pi?i?u?ET????Unknown
u3HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @ar???pi?ib??*????Unknown
b4HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @ar???pi?i??6???Unknown
?5HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ar???pi?iқ???7???Unknown
?6HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @ar???pi?i????Q???Unknown
o7HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @ar???pi?iB?ξ?j???Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aV?	??c?i̾?j?}???Unknown
X9HostEqual"Equal(1      @9      @A      @I      @aV?	??c?iV?B?????Unknown
\:HostGreater"Greater(1      @9      @A      @I      @aV?	??c?i????ƣ???Unknown
e;Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aV?	??c?ij۶m۶???Unknown?
Z<HostSlice"Slice_1(1      @9      @A      @I      @aV?	??c?i??p?????Unknown
?=HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aV?	??c?i~?*?????Unknown
v>HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aV?	??c?i??p????Unknown
v?HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aV?	??c?i??.???Unknown
~@HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aV?	??c?iY?B???Unknown
?AHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aV?	??c?i?tW)???Unknown
}BHostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aV?	??c?i0?l<???Unknown
?CHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aV?	??c?i?'?ˀO???Unknown
?DHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @aV?	??c?iD1Aw?b???Unknown
VEHostCast"Cast(1       @9       @A       @I       @ar???pY?i?7??Mo???Unknown
XFHostCast"Cast_3(1       @9       @A       @I       @ar???pY?i?=9\|???Unknown
XGHostCast"Cast_5(1       @9       @A       @I       @ar???pY?iXD?ξ????Unknown
uHHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @ar???pY?i?J1Aw????Unknown
?IHostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @ar???pY?iQ??/????Unknown
dJHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @ar???pY?ilW)&?????Unknown
jKHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @ar???pY?i?]???????Unknown
rLHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @ar???pY?i$d!Y????Unknown
}MHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @ar???pY?i?j?}????Unknown
`NHostDivNoNan"
div_no_nan(1       @9       @A       @I       @ar???pY?i?p??????Unknown
wOHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @ar???pY?i8w?b?????Unknown
?PHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @ar???pY?i?}?:????Unknown
xQHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @ar???pY?i???G????Unknown
?RHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @ar???pY?iL?	?????Unknown
?SHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @ar???pY?i???,d!???Unknown
?THostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @ar???pY?i??.???Unknown
?UHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @ar???pY?i`?}?:???Unknown
?VHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @ar???pY?i?????G???Unknown
?WHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @ar???pY?i?u?ET???Unknown
~XHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @ar???pY?it??h?`???Unknown
}YHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @ar???pY?iжm۶m???Unknown
ZHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @ar???pY?i,??Moz???Unknown
?[HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @ar???pY?i??e?'????Unknown
v\HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??ar???pI?i?ƣ??????Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??ar???pI?i???2?????Unknown
X^HostCast"Cast_4(1      ??9      ??A      ??I      ??ar???pI?i?l<????Unknown
a_HostIdentity"Identity(1      ??9      ??A      ??I      ??ar???pI?i@?]??????Unknown?
?`HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1      ??9      ??A      ??I      ??ar???pI?inӛ??????Unknown
?aHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??ar???pI?i???Q????Unknown
TbHostMul"Mul(1      ??9      ??A      ??I      ??ar???pI?i??Q?????Unknown
|cHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??ar???pI?i??U?	????Unknown
wdHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??ar???pI?i&???e????Unknown
yeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??ar???pI?iT????????Unknown
?fHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??ar???pI?i??6????Unknown
?gHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??ar???pI?i??Moz????Unknown
?hHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??ar???pI?i?싨?????Unknown
?iHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??ar???pI?i???2????Unknown
?jHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??ar???pI?i:??????Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??ar???pI?ih?ET?????Unknown
?lHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??ar???pI?i????G????Unknown
?mHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??ar???pI?i???ƣ????Unknown
?nHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??ar???pI?i?????????Unknown
voHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor(i?????????Unknown
ipHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
[qHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown
[rHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown2CPU