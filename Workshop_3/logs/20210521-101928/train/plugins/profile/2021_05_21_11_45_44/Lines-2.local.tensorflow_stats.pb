"?h
BHostIDLE"IDLE1    ??@A    ??@a?s3????i?s3?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     X?@9     X?@A     X?@I     X?@a??;Q???i?ڸ-k???Unknown?
iHostWriteSummary"WriteSummary(1     ?@@9     ?@@A     ?@@I     ?@@a?$?*?i?i???ф???Unknown?
vHost_FusedMatMul"sequential_90/dense_279/Relu(1     ?@@9     ?@@A     ?@@I     ?@@a?$?*?i?iY5?v????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1      9@9      9@A      7@I      7@a?捔?a?i@?U????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      4@9      4@A      4@I      4@ad?Z?_?i???]߿???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@a?p???]?i????????Unknown
vHost_FusedMatMul"sequential_90/dense_280/Relu(1      0@9      0@A      0@I      0@a?*??@?X?iQ }\????Unknown
?	HostMatMul".gradient_tape/sequential_90/dense_280/MatMul_1(1      (@9      (@A      (@I      (@a	`???R?i??Td????Unknown
t
HostSigmoid"sequential_90/dense_281/Sigmoid(1      (@9      (@A      (@I      (@a	`???R?i?M?????Unknown
?HostMatMul",gradient_tape/sequential_90/dense_280/MatMul(1      &@9      &@A      &@I      &@a]m?Q?i?_[C????Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@ad?Z?O?i???????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@a????K?i??ٹ???Unknown
dHostDataset"Iterator::Model(1      <@9      <@A      "@I      "@a????K?i-?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      *@9      *@A      "@I      "@a????K?iq
N.???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a????K?i?+?h???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?*??@?H?i?׻?8 ???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?*??@?H?iK??p&???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?*??@?H?i/#Y?,???Unknown?
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?*??@?H?i??V??2???Unknown
?HostMatMul",gradient_tape/sequential_90/dense_281/MatMul(1       @9       @A       @I       @a?*??@?H?i????9???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a_Eٴ??E?i???_?>???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a_Eٴ??E?iN????C???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a_Eٴ??E?i?),gI???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a_Eٴ??E?i?_???N???Unknown
yHost_FusedMatMul"sequential_90/dense_281/BiasAdd(1      @9      @A      @I      @a_Eٴ??E?iA?l?GT???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a	`???B?iW?t?X???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a	`???B?i????]???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a	`???B?i???lDb???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a	`???B?i????f???Unknown
?HostMatMul",gradient_tape/sequential_90/dense_279/MatMul(1      @9      @A      @I      @a	`???B?iyZ.e?k???Unknown
? HostBiasAddGrad"9gradient_tape/sequential_90/dense_281/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a	`???B?iQU?@p???Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @ad?Z???i?fus#t???Unknown
x"HostDataset"#Iterator::Model::ParallelMapV2::Zip(1      G@9      G@A      @I      @ad?Z???i??x???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ad?Z???in????{???Unknown
?$HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @ad?Z???i?H?)????Unknown
?%HostBiasAddGrad"9gradient_tape/sequential_90/dense_280/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ad?Z???i,????????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?*??@?8?ijdɆ???Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?*??@?8?i??*?????Unknown
V(HostMean"Mean(1      @9      @A      @I      @a?*??@?8?i?D? ????Unknown
u)HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?*??@?8?i??]\????Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?*??@?8?i??w8????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?*??@?8?i????S????Unknown
v,HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?*??@?8?iom?To????Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?*??@?8?iTC???????Unknown
?.HostBiasAddGrad"9gradient_tape/sequential_90/dense_279/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?*??@?8?i9ߤ?????Unknown
?/HostMatMul".gradient_tape/sequential_90/dense_281/MatMul_1(1      @9      @A      @I      @a?*??@?8?i??L¢???Unknown
?0HostReadVariableOp".sequential_90/dense_279/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?*??@?8?i??ݥ???Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a	`???2?io%&?2????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a	`???2?iۅ9q?????Unknown
V3HostCast"Cast(1      @9      @A      @I      @a	`???2?iG?L/ܬ???Unknown
X4HostCast"Cast_3(1      @9      @A      @I      @a	`???2?i?F`?0????Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a	`???2?i?s??????Unknown
?6HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      <@9      <@A      @I      @a	`???2?i??iڳ???Unknown
?7HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a	`???2?i?g?'/????Unknown
d8HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a	`???2?icȭ僸???Unknown
?9HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a	`???2?i?(??غ???Unknown
z:HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a	`???2?i;??a-????Unknown
v;HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a	`???2?i????????Unknown
v<HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a	`???2?iJ???????Unknown
?=HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a	`???2?i??+????Unknown
u>HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a	`???2?i?
"Z?????Unknown
~?HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a	`???2?iWk5?????Unknown
?@HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a	`???2?i??H?)????Unknown
?AHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a	`???2?i/,\?~????Unknown
?BHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a	`???2?i??oR?????Unknown
?CHostReluGrad".gradient_tape/sequential_90/dense_279/ReluGrad(1      @9      @A      @I      @a	`???2?i??(????Unknown
?DHostReadVariableOp"-sequential_90/dense_279/MatMul/ReadVariableOp(1      @9      @A      @I      @a	`???2?isM??|????Unknown
?EHostReadVariableOp".sequential_90/dense_281/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a	`???2?i߭???????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?*??@?(?iҘ?`_????Unknown
XGHostCast"Cast_5(1       @9       @A       @I       @a?*??@?(?iŃ?4?????Unknown
XHHostEqual"Equal(1       @9       @A       @I       @a?*??@?(?i?n?{????Unknown
|IHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?*??@?(?i?Y??????Unknown
rJHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?*??@?(?i?D갖????Unknown
}KHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?*??@?(?i?/??$????Unknown
`LHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?*??@?(?i?Y?????Unknown
bMHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?*??@?(?iw-@????Unknown
?NHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?*??@?(?ij??????Unknown
?OHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?*??@?(?i]?*?[????Unknown
?PHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?*??@?(?iP?7??????Unknown
?QHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?*??@?(?iC?D}w????Unknown
?RHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?*??@?(?i6?QQ????Unknown
~SHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?*??@?(?i)?^%?????Unknown
?THostReluGrad".gradient_tape/sequential_90/dense_280/ReluGrad(1       @9       @A       @I       @a?*??@?(?irk? ????Unknown
?UHostReadVariableOp".sequential_90/dense_280/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?*??@?(?i]xͮ????Unknown
?VHostReadVariableOp"-sequential_90/dense_280/MatMul/ReadVariableOp(1       @9       @A       @I       @a?*??@?(?iH??<????Unknown
?WHostReadVariableOp"-sequential_90/dense_281/MatMul/ReadVariableOp(1       @9       @A       @I       @a?*??@?(?i?2?u?????Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a?*??@??in??_?????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a?*??@??i??IX????Unknown?
?ZHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a?*??@??i`??3????Unknown
?[HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor(1      ??9      ??A      ??I      ??a?*??@??i???????Unknown
T\HostMul"Mul(1      ??9      ??A      ??I      ??a?*??@??iR~??????Unknown
s]HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a?*??@??i????s????Unknown
j^HostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a?*??@??iDi??:????Unknown
v_HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a?*??@??i????????Unknown
w`HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?*??@??i6T̯?????Unknown
yaHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?*??@??i??ҙ?????Unknown
xbHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?*??@??i(?كV????Unknown
?cHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?*??@??i???m????Unknown
?dHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?*??@??i*?W?????Unknown
?eHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?*??@??i???A?????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?*??@??i?+r????Unknown
?gHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?*??@??i???9????Unknown
?hHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?*??@??i?????????Unknown
JiHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
WjHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
[kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown*?h
uHostFlushSummaryWriter"FlushSummaryWriter(1     X?@9     X?@A     X?@I     X?@ahL??,??ihL??,???Unknown?
iHostWriteSummary"WriteSummary(1     ?@@9     ?@@A     ?@@I     ?@@aM??l???i?0g?????Unknown?
vHost_FusedMatMul"sequential_90/dense_279/Relu(1     ?@@9     ?@@A     ?@@I     ?@@aM??l???i?????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1      9@9      9@A      7@I      7@aE??????i?*B?]???Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      4@9      4@A      4@I      4@a ??????i"?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@aQE??????i???~S???Unknown
vHost_FusedMatMul"sequential_90/dense_280/Relu(1      0@9      0@A      0@I      0@a?,??4??i???Q????Unknown
?HostMatMul".gradient_tape/sequential_90/dense_280/MatMul_1(1      (@9      (@A      (@I      (@a?a?ߔ???i>??????Unknown
t	HostSigmoid"sequential_90/dense_281/Sigmoid(1      (@9      (@A      (@I      (@a?a?ߔ???i?????O???Unknown
?
HostMatMul",gradient_tape/sequential_90/dense_280/MatMul(1      &@9      &@A      &@I      &@aޮ?wHT??i??a?ߔ???Unknown
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a ?????ip????????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@a???O_[|?i???c????Unknown
dHostDataset"Iterator::Model(1      <@9      <@A      "@I      "@a???O_[|?i???!QE???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      *@9      *@A      "@I      "@a???O_[|?i?~?~???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a???O_[|?i%??????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?,??4y?i^[,(????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?,??4y?i???????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?,??4y?i?F?M???Unknown?
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?,??4y?il??d????Unknown
?HostMatMul",gradient_tape/sequential_90/dense_281/MatMul(1       @9       @A       @I       @a?,??4y?i?4`β???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aIǧ?-v?iU?x??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aIǧ?-v?i??????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aIǧ?-v?is#7r#7???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aIǧ?-v?is???c???Unknown
yHost_FusedMatMul"sequential_90/dense_281/BiasAdd(1      @9      @A      @I      @aIǧ?-v?i???(\????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?a?ߔ?r?iT+?R+????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?a?ߔ?r?i?t|?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?a?ߔ?r?i??3?? ???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?a?ߔ?r?i?e?Ϙ&???Unknown
?HostMatMul",gradient_tape/sequential_90/dense_279/MatMul(1      @9      @A      @I      @a?a?ߔ?r?i`β?gL???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_90/dense_281/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?a?ߔ?r?i#7r#7r???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a ????o?i???????Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1      G@9      G@A      @I      @a ????o?i;?;????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a ????o?i???????Unknown
?#HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a ????o?i???????Unknown
?$HostBiasAddGrad"9gradient_tape/sequential_90/dense_280/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ????o?i???????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?,??4i?i(\???(???Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?,??4i?iU??*B???Unknown
V'HostMean"Mean(1      @9      @A      @I      @a?,??4i?i???O_[???Unknown
u(HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?,??4i?i?-?t???Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?,??4i?i?ȍ?ȍ???Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?,??4i?i	d??????Unknown
v+HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?,??4i?i6??i2????Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?,??4i?ic?0g????Unknown
?-HostBiasAddGrad"9gradient_tape/sequential_90/dense_279/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?,??4i?i?5???????Unknown
?.HostMatMul".gradient_tape/sequential_90/dense_281/MatMul_1(1      @9      @A      @I      @a?,??4i?i???????Unknown
?/HostReadVariableOp".sequential_90/dense_279/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?,??4i?i?k??%???Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?a?ߔ?b?iL k?7???Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?a?ߔ?b?i??J??J???Unknown
V2HostCast"Cast(1      @9      @A      @I      @a?a?ߔ?b?i?*B?]???Unknown
X3HostCast"Cast_3(1      @9      @A      @I      @a?a?ߔ?b?ir=
ףp???Unknown
\4HostGreater"Greater(1      @9      @A      @I      @a?a?ߔ?b?i???k?????Unknown
?5HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      <@9      <@A      @I      @a?a?ߔ?b?i6?? s????Unknown
?6HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?a?ߔ?b?i?Z??Z????Unknown
d7HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a?a?ߔ?b?i??*B????Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?a?ߔ?b?i\?h?)????Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?a?ߔ?b?i?wHT????Unknown
v:HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?a?ߔ?b?i ,(??????Unknown
v;HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?a?ߔ?b?i??~????Unknown
?<HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a?a?ߔ?b?i???????Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?a?ߔ?b?iFIǧ?-???Unknown
~>HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?a?ߔ?b?i???<?@???Unknown
??HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?a?ߔ?b?i
???~S???Unknown
?@HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?a?ߔ?b?ilfffff???Unknown
?AHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?a?ߔ?b?i?F?My???Unknown
?BHostReluGrad".gradient_tape/sequential_90/dense_279/ReluGrad(1      @9      @A      @I      @a?a?ߔ?b?i0?%?5????Unknown
?CHostReadVariableOp"-sequential_90/dense_279/MatMul/ReadVariableOp(1      @9      @A      @I      @a?a?ߔ?b?i??%????Unknown
?DHostReadVariableOp".sequential_90/dense_281/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?a?ߔ?b?i?7??????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?,??4Y?i?%?????Unknown
XFHostCast"Cast_5(1       @9       @A       @I       @a?,??4Y?i ?d?9????Unknown
XGHostEqual"Equal(1       @9       @A       @I       @a?,??4Y?i?????????Unknown
|HHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?,??4Y?iLn?Fn????Unknown
rIHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?,??4Y?i?;$?????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?,??4Y?ix	d?????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?,??4Y?iףp=
???Unknown
bLHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?,??4Y?i????????Unknown
?MHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?,??4Y?i:r#7r#???Unknown
?NHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?,??4Y?i??c?0???Unknown
?OHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?,??4Y?if???<???Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?,??4Y?i???`AI???Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?,??4Y?i??"??U???Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?,??4Y?i(vb'vb???Unknown
?SHostReluGrad".gradient_tape/sequential_90/dense_280/ReluGrad(1       @9       @A       @I       @a?,??4Y?i?C??o???Unknown
?THostReadVariableOp".sequential_90/dense_280/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?,??4Y?iT???{???Unknown
?UHostReadVariableOp"-sequential_90/dense_280/MatMul/ReadVariableOp(1       @9       @A       @I       @a?,??4Y?i??!QE????Unknown
?VHostReadVariableOp"-sequential_90/dense_281/MatMul/ReadVariableOp(1       @9       @A       @I       @a?,??4Y?i??a?ߔ???Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??a?,??4I?iK??,????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??a?,??4I?iz?z????Unknown?
?YHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a?,??4I?i?`AIǧ???Unknown
?ZHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor(1      ??9      ??A      ??I      ??a?,??4I?i?G?z????Unknown
T[HostMul"Mul(1      ??9      ??A      ??I      ??a?,??4I?iw.??a????Unknown
s\HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a?,??4I?iB!ޮ????Unknown
j]HostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a?,??4I?i???????Unknown
v^HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a?,??4I?i??`AI????Unknown
w_HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?,??4I?i?? s?????Unknown
y`HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?,??4I?in????????Unknown
xaHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?,??4I?i9?@?0????Unknown
?bHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?,??4I?i~?~????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?,??4I?i?d?9?????Unknown
?dHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?,??4I?i?K k????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?,??4I?ie2??e????Unknown
?fHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?,??4I?i0`β????Unknown
?gHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?,??4I?i?????????Unknown
JhHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
WiHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
[jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU