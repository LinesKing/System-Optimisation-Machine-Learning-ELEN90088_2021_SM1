"?e
BHostIDLE"IDLE1     8?@A     8?@a? ?r???i? ?r????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?Y@9     ?Y@A     @X@I     @X@a{?C"n???i_?}]c???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a????m`??iN???`????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      I@9      I@A      I@I      I@a=iXdy??i?0T|FP???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?A@9     ?A@A     ?A@I     ?A@a?ɬ=?!??i?J?͔???Unknown
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1     ?@@9     ?@@A     ?@@I     ?@@a????('??i??7j????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?@@9     ?@@A      @@I      @@aN ???S?i>????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      .@9      .@A      .@I      .@aI~?ix^m?i??~p1???Unknown
V	HostCast"Cast(1      (@9      (@A      (@I      (@a;????~g?iT?9??H???Unknown
i
HostWriteSummary"WriteSummary(1      (@9      (@A      (@I      (@a;????~g?i????m`???Unknown?
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      (@9      (@A      (@I      (@a;????~g?i???k?w???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      &@9      &@A      &@I      &@a6?????e?izu0?u????Unknown
gHostStridedSlice"strided_slice(1      &@9      &@A      &@I      &@a6?????e?ip1???????Unknown
dHostDataset"Iterator::Model(1     @T@9     @T@A      $@I      $@a1T|FP?c?iĭ?ғ????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a1T|FP?c?i*>#(????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a1T|FP?c?il??s?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a,?<?a?i㐈[????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      "@9      "@A      "@I      "@a,?<?a?i???? ???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      "@9      "@A      "@I      "@a,?<?a?i?\??????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aN ???S_?i?Y{?C"???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aN ???S_?i?VMf?1???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @aN ???S_?i?S@?A???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @aN ???S_?i?P?AQ???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aE?z/=i[?i0???^???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aE?z/=i[?i?? W?l???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aE?z/=i[?i???^z???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aE?z/=i[?izFP?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a;????~W?iFĭ?ғ???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a;????~W?iB[?????Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a;????~W?i޿h?Q????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a;????~W?i?=?!????Unknown
e Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a1T|FP?S?i?{?I?????Unknown?
v!HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a1T|FP?S?i??r?????Unknown
~"HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a1T|FP?S?i(?/?o????Unknown
t#Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a1T|FP?S?iR6S?9????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aN ???SO?i?4<?????Unknown
\%HostGreater"Greater(1      @9      @A      @I      @aN ???SO?ib3%??????Unknown
V&HostMean"Mean(1      @9      @A      @I      @aN ???SO?i?1??????Unknown
?'HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aN ???SO?ir0?u?????Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aN ???SO?i?.?bb???Unknown
z)HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aN ???SO?i?-?O7???Unknown
b*HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aN ???SO?i
,?<???Unknown
?+HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aN ???SO?i?*?)????Unknown
?,HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aN ???SO?i)??$???Unknown
o-HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @aN ???SO?i?'m?,???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a;????~G?i???j2???Unknown
X/HostCast"Cast_3(1      @9      @A      @I      @a;????~G?in??fJ8???Unknown
X0HostEqual"Equal(1      @9      @A      @I      @a;????~G?iTdy*>???Unknown
?1HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Z@9     ?Z@A      @I      @a;????~G?i:#(?	D???Unknown
?2HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a;????~G?i ??{?I???Unknown
v3HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a;????~G?i??-?O???Unknown
~4HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a;????~G?i?_4ߨU???Unknown
v5HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a;????~G?i?㐈[???Unknown
`6HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a;????~G?i?ݑBha???Unknown
?7HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a;????~G?i??@?Gg???Unknown
?8HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a;????~G?i?[??'m???Unknown
?9HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a;????~G?ij?Ws???Unknown
}:HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a;????~G?iP?L	?x???Unknown
?;HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a;????~G?i6????~???Unknown
?<HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a;????~G?iW?l?????Unknown
t=HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aN ???S??i`?㐈???Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aN ???S??i?U?Y{????Unknown
|?HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aN ???S??i???e????Unknown
d@HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aN ???S??i,T|FP????Unknown
jAHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aN ???S??ip???:????Unknown
rBHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aN ???S??i?Re3%????Unknown
vCHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aN ???S??i??٩????Unknown
vDHostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @aN ???S??i<QN ?????Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aN ???S??i???????Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aN ???S??i?O7ϫ???Unknown
?GHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @aN ???S??iϫ??????Unknown
?HHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1       @9       @A       @I       @aN ???S??iLN ??????Unknown
?IHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aN ???S??i?͔p?????Unknown
?JHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aN ???S??i?L	?x????Unknown
?KHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @aN ???S??i?}]c????Unknown
~LHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aN ???S??i\K??M????Unknown
}MHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aN ???S??i??fJ8????Unknown
NHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @aN ???S??i?I??"????Unknown
?OHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aN ???S??i(?O7????Unknown
?PHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aN ???S??ilHĭ?????Unknown
?QHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @aN ???S??i??8$?????Unknown
?RHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aN ???S??i?F???????Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aN ???S/?i?????????Unknown
XTHostCast"Cast_4(1      ??9      ??A      ??I      ??aN ???S/?i8?!?????Unknown
XUHostCast"Cast_5(1      ??9      ??A      ??I      ??aN ???S/?i?\L?????Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??aN ???S/?i|E???????Unknown?
?WHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??aN ???S/?i??????Unknown
TXHostMul"Mul(1      ??9      ??A      ??I      ??aN ???S/?i??
??????Unknown
uYHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??aN ???S/?ibE9?????Unknown
?ZHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??aN ???S/?iDtv????Unknown
}[HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aN ???S/?i????k????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aN ???S/?iH???`????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aN ???S/?i?.&V????Unknown
?^HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??aN ???S/?i?BhaK????Unknown
x_HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??aN ???S/?i.???@????Unknown
?`Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aN ???S/?i????5????Unknown
?aHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aN ???S/?ir+????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aN ???S/?iAQN ????Unknown
?cHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aN ???S/?i????????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aN ???S/?iX???
????Unknown
?eHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??aN ???S/?i?????????Unknown
WfHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown
]iHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i?????????Unknown*?d
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?Y@9     ?Y@A     @X@I     @X@a???d????i???d?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a@l*9???i???Q????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      I@9      I@A      I@I      I@a?$I?$I??i??Tr????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?A@9     ?A@A     ?A@I     ?A@a????????i??M?!???Unknown
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1     ?@@9     ?@@A     ?@@I     ?@@a?NV?#??i????&???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?@@9     ?@@A      @@I      @@a??4??g??i?jyc???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      .@9      .@A      .@I      .@a_?_???i???Tr???Unknown
VHostCast"Cast(1      (@9      (@A      (@I      (@aE'?卑?i?????????Unknown
i	HostWriteSummary"WriteSummary(1      (@9      (@A      (@I      (@aE'?卑?iQ???Q???Unknown?
?
Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      (@9      (@A      (@I      (@aE'?卑?iK?w?Z????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      &@9      &@A      &@I      &@a?4??g??i?_?_???Unknown
gHostStridedSlice"strided_slice(1      &@9      &@A      &@I      &@a?4??g??i?7F0?????Unknown
dHostDataset"Iterator::Model(1     @T@9     @T@A      $@I      $@a?A?A??i?>???T???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?A?A??i5F0??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?A?A??i?M?!?>???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a?>???T??i?:??:????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      "@9      "@A      "@I      "@a?>???T??i}'??????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      "@9      "@A      "@I      "@a?>???T??iy?G?z???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??4??g??i ?>??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a??4??g??iǹ?. 6???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a??4??g??in?`??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??4??g??i_?_????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a{?G?z??ig?JC???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a{?G?z??i??. 6????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a{?G?z??i?M?!????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a{?G?z??i]@l*9???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aE'?十?iZ??D???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aE'?十?iW|?W|????Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aE'?十?iTr?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @aE'?十?iQ???Q???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?A?A}?i?;Y-o????Unknown?
v HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?A?A}?i?????????Unknown
~!HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?A?A}?iIC?}v???Unknown
t"Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a?A?A}?i??&?;???Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??4??gw?iD0???j???Unknown
\$HostGreater"Greater(1      @9      @A      @I      @a??4??gw?i?????????Unknown
V%HostMean"Mean(1      @9      @A      @I      @a??4??gw?i?bSi????Unknown
?&HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??4??gw?i=l*9????Unknown
?'HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??4??gw?i????&???Unknown
z(HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??4??gw?i?>???T???Unknown
b)HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??4??gw?i6??:?????Unknown
?*HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??4??gw?i?L?w????Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??4??gw?i?z?G????Unknown
o,HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a??4??gw?i/??g???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aE'???q?i.33333???Unknown
X.HostCast"Cast_3(1      @9      @A      @I      @aE'???q?i-???NV???Unknown
X/HostEqual"Equal(1      @9      @A      @I      @aE'???q?i,???jy???Unknown
?0HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Z@9     ?Z@A      @I      @aE'???q?i+ 6??????Unknown
?1HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aE'???q?i*o?`?????Unknown
v2HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aE'???q?i)??+?????Unknown
~3HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aE'???q?i(9?????Unknown
v4HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aE'???q?i'\???(???Unknown
`5HostDivNoNan"
div_no_nan(1      @9      @A      @I      @aE'???q?i&???L???Unknown
?6HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aE'???q?i%?;Y-o???Unknown
?7HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aE'???q?i$I?$I????Unknown
?8HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aE'???q?i#???d????Unknown
}9HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aE'???q?i"?>??????Unknown
?:HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @aE'???q?i!6???????Unknown
?;HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @aE'???q?i ??Q????Unknown
t<HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??4??gg?iʹ?. 6???Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??4??gg?it???M???Unknown
|>HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??4??gg?i#???d???Unknown
d?HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??4??gg?i?W|?W|???Unknown
j@HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??4??gg?ir?`??????Unknown
rAHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??4??gg?i?D'????Unknown
vBHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??4??gg?i??(\?????Unknown
vCHostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a??4??gg?ip*9?????Unknown
uDHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??4??gg?i_?_????Unknown
wEHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??4??gg?iē??????Unknown
?FHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??4??gg?inȹ?. ???Unknown
?GHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1       @9       @A       @I       @a??4??gg?i????7???Unknown
?HHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??4??gg?i?1???N???Unknown
?IHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??4??gg?ilfffff???Unknown
?JHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a??4??gg?i?JC?}???Unknown
~KHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??4??gg?i??. 6????Unknown
}LHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a??4??gg?ij??????Unknown
MHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a??4??gg?i9??????Unknown
?NHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??4??gg?i?m۶m????Unknown
?OHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??4??gg?ih????????Unknown
?PHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a??4??gg?iףp=
???Unknown
?QHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??4??gg?i??M?!???Unknown
vRHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??4??gW?i&?;Y-???Unknown
XSHostCast"Cast_4(1      ??9      ??A      ??I      ??a??4??gW?if@l*9???Unknown
XTHostCast"Cast_5(1      ??9      ??A      ??I      ??a??4??gW?i?Z??D???Unknown
aUHostIdentity"Identity(1      ??9      ??A      ??I      ??a??4??gW?iuPuP???Unknown?
?VHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??4??gW?ie???(\???Unknown
TWHostMul"Mul(1      ??9      ??A      ??I      ??a??4??gW?i??4??g???Unknown
uXHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??4??gW?iĦҐs???Unknown
?YHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??4??gW?id??D???Unknown
}ZHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??4??gW?i?????????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??4??gW?i???????Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??4??gW?ic-o?`????Unknown
?]HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a??4??gW?i?G?z????Unknown
x^HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a??4??gW?ibSiȹ???Unknown
?_Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??4??gW?ib|?W|????Unknown
?`HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??4??gW?i??7F0????Unknown
?aHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??4??gW?i??4?????Unknown
?bHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??4??gW?ia?#?????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??4??gW?i???L????Unknown
?dHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a??4??gW?i     ???Unknown
WeHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i     ???Unknown
YfHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown
]hHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i     ???Unknown2CPU