"?b
BHostIDLE"IDLE1     ?@A     ?@au	?d??iu	?d???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?^@9     ?^@A     ?^@I     ?^@a?^?"?'??i?z8?g???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?X@9     ?X@A     ?X@I     ?X@a??Æ????i??n?5???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1     ?@@9     ?@@A     ?@@I     ?@@aFF?+?V??i???jz???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      <@9      <@A      <@I      <@aĆ??l}?iǺ?C????Unknown?
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      7@9      7@A      7@I      7@a???%+x?i??H?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      5@9      5@A      5@I      5@a??Nv?i??o????Unknown
jHostMean"binary_crossentropy/Mean(1      .@9      .@A      .@I      .@a?"X??o?i?Y?A1???Unknown
d	HostDataset"Iterator::Model(1      a@9      a@A      ,@I      ,@aĆ??lm?i=?פ?N???Unknown
`
HostDivNoNan"
div_no_nan(1      (@9      (@A      (@I      (@a?N8i?i????g???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?ce?i??L??|???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a?ce?i?#???????Unknown?
^HostGatherV2"GatherV2(1       @9       @A       @I       @aK???`?i?0f???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @aK???`?ix=??????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @aK???`?iWJ?^????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @aK???`?i6W?.????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @aK???`?id>)?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aĆ??l]?iX?6?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aĆ??l]?i???Bj???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aĆ??l]?i?E?O ???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @aĆ??l]?i!??\? ???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aĆ??l]?idܚi?/???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aĆ??l]?i?'zvB>???Unknown
\HostGreater"Greater(1      @9      @A      @I      @a?N8Y?iN????J???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?N8Y?i?:??zW???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?N8Y?i?ē?d???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?N8Y?iCN???p???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?N8Y?i?פ?N}???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?N8Y?i?a???????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a?N8Y?i8??Æ????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a?N8Y?i?t??"????Unknown
V HostMean"Mean(1      @9      @A      @I      @a?cU?i?<?פ????Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @a?cU?i?"?&????Unknown
|"HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?cU?i?S??????Unknown
a#HostCast"sequential/Cast(1      @9      @A      @I      @a?cU?i???*????Unknown
?$HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?cU?i]???????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aK???P?i?c????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aK???P?i?im}????Unknown
e'Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aK???P?ikp??????Unknown?
s(HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aK???P?i?v#M????Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aK???P?iK}~!????Unknown
?*HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aK???P?i???(
???Unknown
t+Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aK???P?i+?40????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?N8I?i?θ5????Unknown
u-HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?N8I?i?=;!???Unknown
?.HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?N8I?i?X?@o%???Unknown
?/HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?N8I?i{?EF?+???Unknown
z0HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?N8I?iO??K2???Unknown
~1HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?N8I?i#'NQY8???Unknown
v2HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?N8I?i?k?V?>???Unknown
b3HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?N8I?i˰V\?D???Unknown
?4HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?N8I?i???aCK???Unknown
?5HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?N8I?is:_g?Q???Unknown
?6HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      @9      @A      @I      @a?N8I?iG?l?W???Unknown
?7Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?N8I?i?gr-^???Unknown
?8HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?N8I?i??w{d???Unknown
?9HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?N8I?i?Mp}?j???Unknown
?:HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?N8I?i????q???Unknown
?;HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?N8I?ik?x?ew???Unknown
?<HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a?N8I?i????}???Unknown
t=HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aK???@?iw????????Unknown
V>HostCast"Cast(1       @9       @A       @I       @aK???@?i?"X?????Unknown
X?HostCast"Cast_2(1       @9       @A       @I       @aK???@?i???O????Unknown
X@HostEqual"Equal(1       @9       @A       @I       @aK???@?i)???????Unknown
TAHostMul"Mul(1       @9       @A       @I       @aK???@?iW?`??????Unknown
|BHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aK???@?i?/??????Unknown
dCHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aK???@?iǲ??????Unknown
rDHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aK???@?i?5i?S????Unknown
vEHostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @aK???@?i7???????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aK???@?io<Ĳ?????Unknown
}GHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aK???@?i??q??????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aK???@?i?B?#????Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aK???@?i?̽W????Unknown
xJHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aK???@?iOIz??????Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @aK???@?i??'ſ????Unknown
?LHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @aK???@?i?O???????Unknown
?MHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aK???@?i?҂?'????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aK???@?i/V0?[????Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aK???@?ig??ӏ????Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aK???@?i?\???????Unknown
?QHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aK???@?i??8??????Unknown
?RHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @aK???@?ic??+????Unknown
oSHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @aK???@?iG???_????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aK???0?i??j?y????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??aK???0?iiA??????Unknown
XVHostCast"Cast_3(1      ??9      ??A      ??I      ??aK???0?i+??????Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??aK???0?i?????????Unknown?
vXHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??aK???0?iS????????Unknown
?YHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??aK???0?i?o???????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aK???0?i?1s?????Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aK???0?i'?I?/????Unknown
?\HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aK???0?iô ?I????Unknown
?]HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aK???0?i_v??c????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aK???0?i?7??}????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aK???0?i?????????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aK???0?i3?{??????Unknown
}aHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??aK???0?i?|R??????Unknown
bHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??aK???0?ik>)??????Unknown
?cHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aK???0?i     ???Unknown
+dHostCast"Cast_4(i     ???Unknown
[eHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i     ???Unknown
YfHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown*?b
sHostDataset"Iterator::Model::ParallelMapV2(1     ?^@9     ?^@A     ?^@I     ?^@aqy4\??iqy4\???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?X@9     ?X@A     ?X@I     ?X@a&?vI?]??iL??Ri????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1     ?@@9     ?@@A     ?@@I     ?@@a???b輨?i~4_????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      <@9      <@A      <@I      <@aP??S`???i?ɲi?l???Unknown?
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      7@9      7@A      7@I      7@aa?D?=??iʵQrm????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      5@9      5@A      5@I      5@a?A?}|??i??0z.????Unknown
jHostMean"binary_crossentropy/Mean(1      .@9      .@A      .@I      .@ah??Y0}??i ?????Unknown
dHostDataset"Iterator::Model(1      a@9      a@A      ,@I      ,@aP??S`???i
????????Unknown
`	HostDivNoNan"
div_no_nan(1      (@9      (@A      (@I      (@a ?G????ig???1???Unknown
l
HostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a??w@???i+??ʩ???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a??w@???i"??Ȼ!???Unknown?
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??_ ???i(?ʯ????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a??_ ???i.??ˣ????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??_ ???i4_͗A???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a??_ ???i:/?΋????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a??_ ???i@?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aP??S`???iE?UQuU???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aP??S`???iJ???j????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aP??S`???iO??S`????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @aP??S`???iTWE?UQ???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aP??S`???iY-?VK????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aP??S`???i^??@????Unknown
\HostGreater"Greater(1      @9      @A      @I      @a ?G????ib??7A???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?G????if?$?.????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?G????ij?D?%????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?G????insd????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?G????irO??a???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a ?G????iv+??
????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a ?G????iz??????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a ?G????i~????8???Unknown
VHostMean"Mean(1      @9      @A      @I      @a??w@?}?i???a?t???Unknown
V HostSum"Sum_2(1      @9      @A      @I      @a??w@?}?i?????????Unknown
|!HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??w@?}?i???c?????Unknown
a"HostCast"sequential/Cast(1      @9      @A      @I      @a??w@?}?i?k???(???Unknown
?#HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??w@?}?i?M?e?d???Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??_ ?w?i?5Sf͔???Unknown
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??_ ?w?i?g?????Unknown
e&Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a??_ ?w?i??g?????Unknown?
s'HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??_ ?w?i???h?$???Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??_ ?w?i??Ri?T???Unknown
?)HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??_ ?w?i??j?????Unknown
t*Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a??_ ?w?i???j?????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a ?G??q?i??b??????Unknown
u,HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a ?G??q?i???k?????Unknown
?-HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?G??q?i?o??? ???Unknown
?.HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a ?G??q?i?]m?D???Unknown
z/HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a ?G??q?i?K???h???Unknown
~0HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a ?G??q?i?92n?????Unknown
v1HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a ?G??q?i?'????Unknown
b2HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a ?G??q?i?Ro?????Unknown
?3HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a ?G??q?i????????Unknown
?4HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a ?G??q?i??qp|???Unknown
?5HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      @9      @A      @I      @a ?G??q?i???w@???Unknown
?6Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a ?G??q?i?͑qsd???Unknown
?7HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a ?G??q?i??!?n????Unknown
?8HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ?G??q?ié?rj????Unknown
?9HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ?G??q?iŗA?e????Unknown
?:HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a ?G??q?iǅ?sa????Unknown
?;HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a ?G??q?i?sa?\???Unknown
t<HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??_ ?g?i?g??Y0???Unknown
V=HostCast"Cast(1       @9       @A       @I       @a??_ ?g?i?[!?VH???Unknown
X>HostCast"Cast_2(1       @9       @A       @I       @a??_ ?g?i?O??S`???Unknown
X?HostEqual"Equal(1       @9       @A       @I       @a??_ ?g?i?C??Px???Unknown
T@HostMul"Mul(1       @9       @A       @I       @a??_ ?g?i?7A?M????Unknown
|AHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??_ ?g?i?+??J????Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??_ ?g?i??G????Unknown
rCHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??_ ?g?i?a?D????Unknown
vDHostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a??_ ?g?i???A????Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??_ ?g?i?? ?>???Unknown
}FHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??_ ?g?i????; ???Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??_ ?g?i????88???Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??_ ?g?i??@?5P???Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??_ ?g?i?ˠ?2h???Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a??_ ?g?i?? ?/????Unknown
?KHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a??_ ?g?i??`?,????Unknown
?LHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??_ ?g?i????)????Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??_ ?g?i?? ?&????Unknown
~NHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??_ ?g?i?#????Unknown
?OHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??_ ?g?i???? ????Unknown
?PHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??_ ?g?i?w@????Unknown
?QHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a??_ ?g?i?k??(???Unknown
oRHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @a??_ ?g?i?_ ?@???Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??_ ?W?i?Y0}L???Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??_ ?W?i?S`?X???Unknown
XUHostCast"Cast_3(1      ??9      ??A      ??I      ??a??_ ?W?i?M?}d???Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??a??_ ?W?i?G??p???Unknown?
vWHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a??_ ?W?i?A?}|???Unknown
?XHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??_ ?W?i?; ?????Unknown
wYHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??_ ?W?i?5P~????Unknown
yZHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??_ ?W?i?/??????Unknown
?[HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??_ ?W?i?)?~
????Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??_ ?W?i $??????Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??_ ?W?i????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??_ ?W?i@?????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??_ ?W?ip????Unknown
}`HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a??_ ?W?i??????Unknown
aHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a??_ ?W?i?????Unknown
?bHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??_ ?W?i     ???Unknown
+cHostCast"Cast_4(i     ???Unknown
[dHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i     ???Unknown
YeHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown2CPU