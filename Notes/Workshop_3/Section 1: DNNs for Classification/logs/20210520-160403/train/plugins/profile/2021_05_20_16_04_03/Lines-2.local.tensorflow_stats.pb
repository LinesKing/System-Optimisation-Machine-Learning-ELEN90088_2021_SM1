"?f
BHostIDLE"IDLE1     ??@A     ??@a? ф}Z??i? ф}Z???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      U@9      U@A      U@I      U@a?P????ie??b????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      J@9      J@A     ?E@I     ?E@aW	`3???i?cw?????Unknown
dHostDataset"Iterator::Model(1     @^@9     @^@A     ?B@I     ?B@a???aՙ??i]???w????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?U@9     ?U@A      @@I      @@a?w????i?f 52$???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@a	`3???i@4a??l???Unknown?
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      :@9      :@A      :@I      :@aR??m?[~?iQ?<?M????Unknown
^HostGatherV2"GatherV2(1      1@9      1@A      1@I      1@a]???w?s?if?.? ????Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(1      1@9      1@A      1@I      1@a]???w?s?i{? г????Unknown
`
HostGatherV2"
GatherV2_1(1      0@9      0@A      0@I      0@a?w??r?i??1????Unknown
VHostSum"Sum_2(1      (@9      (@A      (@I      (@a8????l?iRR??:???Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@a£??i?i?i??S???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      &@9      &@A      &@I      &@a£??i?i֙?vm???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@aߔ?2Zg?i?.? ф???Unknown?
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a????`e?i???aՙ???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a????`e?i?:??ٮ???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a????`e?i???#?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?w??b?i?7???????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?w??b?iۮ?A;????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a?w??b?i?%???????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a?w??b?i??_????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?5h'?X`?iC?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?5h'?X`?iym?I/???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?5h'?X`?i??4??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?5h'?X`?i?=\T?O???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a?5h'?X`?i??T`???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a8????\?in???Vn???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aߔ?2ZW?i?I/z???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?w??R?ij??][????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?w??R?i??7??????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?w??R?i????	????Unknown?
? HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?w??R?i8@4a????Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?w??R?i?s?{?????Unknown
?"HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?w??R?i&?H?????Unknown
~#HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?w??R?i???
g????Unknown
?$HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?w??R?i>&QR?????Unknown
?%HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?w??R?i?aՙ????Unknown
?&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?w??R?iV?Y?l????Unknown
t'HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a8????L?i???Vn????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @a8????L?i????o????Unknown
?)HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?K@9     ?K@A      @I      @a8????L?iQ#CBq????Unknown
V*HostMean"Mean(1      @9      @A      @I      @a8????L?i?O??r????Unknown
r+HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a8????L?i?|?-t????Unknown
?,HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a8????L?iL?,?u???Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a8????L?i???w???Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a8????L?i?s?x???Unknown
?/HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a8????L?iG/z???Unknown
?0Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a8????L?i?[?y{???Unknown
?1HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a8????L?i??\?|$???Unknown
}2HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a8????L?iB??d~+???Unknown
?3HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a8????L?i????2???Unknown
?4HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a8????L?i?FP?9???Unknown
?5HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a8????L?i=;?ł@???Unknown
?6HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a8????L?i?g?;?G???Unknown
t7Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a8????L?i??/??N???Unknown
?8HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a8????L?i8??&?U???Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?w??B?i?ޔ?2Z???Unknown
V:HostCast"Cast(1       @9       @A       @I       @a?w??B?i??Vn?^???Unknown
X;HostCast"Cast_3(1       @9       @A       @I       @a?w??B?i??c???Unknown
s<HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?w??B?iP8۵5h???Unknown
u=HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a?w??B?iV?Y?l???Unknown
|>HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?w??B?i?s_??q???Unknown
d?HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?w??B?i??!?8v???Unknown
j@HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?w??B?ih??D?z???Unknown
zAHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?w??B?i.ͥ?????Unknown
vBHostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a?w??B?i??g?;????Unknown
vCHostMul"%binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a?w??B?i?*0?????Unknown
vDHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?w??B?i?&?Ӓ????Unknown
?EHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?w??B?iFD?w>????Unknown
`FHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?w??B?ibp?????Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?w??B?i?2??????Unknown
bHHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?w??B?i???bA????Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?w??B?i^???????Unknown
?JHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?w??B?i$?x??????Unknown
xKHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?w??B?i??:ND????Unknown
~LHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?w??B?i????????Unknown
?MHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?w??B?iv2???????Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?w??B?i<P?9G????Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?w??B?inC??????Unknown
?PHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?w??B?iȋ??????Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?w??B?i???$J????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?w??B?iTǉ??????Unknown
SHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?w??B?i?Kl?????Unknown
?THostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?w??B?i?M????Unknown
oUHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @a?w??B?i? г?????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?w??2?i?/??N????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?w??2?il>?W?????Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a?w??2?iOMs)?????Unknown
XYHostCast"Cast_5(1      ??9      ??A      ??I      ??a?w??2?i2\T?O????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a?w??2?ik5ͥ????Unknown?
T[HostMul"Mul(1      ??9      ??A      ??I      ??a?w??2?i?y??????Unknown
v\HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a?w??2?iۈ?pQ????Unknown
}]HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?w??2?i???B?????Unknown
w^HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?w??2?i????????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?w??2?i????R????Unknown
?`HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?w??2?ig?{??????Unknown
?aHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?w??2?iJ?\??????Unknown
?bHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?w??2?i-?=\T????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?w??2?i?.?????Unknown
?dHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?w??2?i?????????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?w??2?ik???*???Unknown
?fHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?w??2?i???U???Unknown
?gHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?w??2?iO?Ѻ????Unknown
}hHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a?w??2?i?£????Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?£????Unknown*?e
sHostDataset"Iterator::Model::ParallelMapV2(1      U@9      U@A      U@I      U@a?E(B??i?E(B???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      J@9      J@A     ?E@I     ?E@a?Ո?Y???i??C????Unknown
dHostDataset"Iterator::Model(1     @^@9     @^@A     ?B@I     ?B@a04U_h??i{l???????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?U@9     ?U@A      @@I      @@a7& nL??iB?P?"???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@a??
z??i??k??Q???Unknown?
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      :@9      :@A      :@I      :@a?
z^??i???Ո????Unknown
^HostGatherV2"GatherV2(1      1@9      1@A      1@I      1@a??5???iO??%?????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      1@9      1@A      1@I      1@a??5???iٽ?u?{???Unknown
`	HostGatherV2"
GatherV2_1(1      0@9      0@A      0@I      0@a7& nL??i<??Wv ???Unknown
V
HostSum"Sum_2(1      (@9      (@A      (@I      (@a??	?R???i??.???Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@aF
	????iU_h?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      &@9      &@A      &@I      &@aF
	????i??|NO???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a?7??o??if鸢?????Unknown?
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a???????i?$I?$I???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a???????i?_ف|????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a???????i??iq?5???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a7& nL??i???)????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a7& nL??ij?7???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a7& nL??i?8??iq???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a7& nL??iNmjS?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\???iT??Ԧ6???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\???iZ?JV?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\???i`??׽????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ap?\???if%+Y?J???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @ap?\???ilS??Ԧ???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a??	?R???i?z?$?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?7??o??i??K8y7???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a7& nLz?i??l???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a7& nLz?iW????????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a7& nLz?i???C????Unknown?
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a7& nLz?i?L??	???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a7& nLz?i;??u>???Unknown
?!HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a7& nLz?i?9?as???Unknown
~"HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a7& nLz?i?S>?????Unknown
?#HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a7& nLz?inL@????Unknown
?$HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a7& nLz?ik???????Unknown
?%HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a7& nLz?i????qE???Unknown
t&HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??	?R?s?ip??w?l???Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a??	?R?s?i)?,W????Unknown
?(HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?K@9     ?K@A      @I      @a??	?R?s?i??\?ɻ???Unknown
V)HostMean"Mean(1      @9      @A      @I      @a??	?R?s?i???g<????Unknown
r*HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a??	?R?s?iT??
???Unknown
?+HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??	?R?s?i??!2???Unknown
?,HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??	?R?s?i?,W?Y???Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??	?R?s?i@M?????Unknown
?.HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??	?R?s?i8T}?y????Unknown
?/Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??	?R?s?i?g?F?????Unknown
?0HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??	?R?s?i?{??^????Unknown
}1HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a??	?R?s?ic??????Unknown
?2HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?i?=6DF???Unknown
?3HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?iնm۶m???Unknown
?4HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?i?ʝ?)????Unknown
?5HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?iG??%?????Unknown
t6Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a??	?R?s?i ???????Unknown
?7HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??	?R?s?i?.p????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a7& nLj?i?N??%???Unknown
V9HostCast"Cast(1       @9       @A       @I       @a7& nLj?i nL@???Unknown
X:HostCast"Cast_3(1       @9       @A       @I       @a7& nLj?i+-??fZ???Unknown
s;HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a7& nLj?iQ:?(?t???Unknown
u<HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a7& nLj?iwGΖ?????Unknown
|=HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a7& nLj?i?T?L????Unknown
d>HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a7& nLj?i?as?????Unknown
j?HostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a7& nLj?i?n.??????Unknown
z@HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a7& nLj?i|NO1????Unknown
vAHostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a7& nLj?i5?n?}???Unknown
vBHostMul"%binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a7& nLj?i[??+?,???Unknown
vCHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a7& nLj?i????G???Unknown
?DHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a7& nLj?i???ca???Unknown
`EHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a7& nLj?iͽ?u?{???Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i????????Unknown
bGHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a7& nLj?i?.RH????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i??N??????Unknown
?IHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a7& nLj?ie?n.?????Unknown
xJHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a7& nLj?i????-????Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a7& nLj?i??
z???Unknown
?LHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a7& nLj?i??x?3???Unknown
?MHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a7& nLj?i?&??N???Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a7& nLj?i#4U_h???Unknown
?OHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a7& nLj?iIA/ë????Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a7& nLj?ioNO1?????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a7& nLj?i?[o?D????Unknown
RHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a7& nLj?i?h??????Unknown
?SHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a7& nLj?i?u?{?????Unknown
oTHostSigmoid"sequential/dense_2/Sigmoid(1       @9       @A       @I       @a7& nLj?i???)???Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a7& nLZ?i??? P???Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a7& nLZ?i-??Wv ???Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??a7& nLZ?i?????-???Unknown
XXHostCast"Cast_5(1      ??9      ??A      ??I      ??a7& nLZ?iS???:???Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a7& nLZ?i????G???Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??a7& nLZ?iy?/4U???Unknown
v[HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a7& nLZ?i??k5b???Unknown
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a7& nLZ?i??O?[o???Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7& nLZ?i2?_ف|???Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7& nLZ?i??o?????Unknown
?_HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a7& nLZ?iX?GΖ???Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a7& nLZ?i?я~?????Unknown
?aHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a7& nLZ?i~؟?????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a7& nLZ?i߯?@????Unknown
?cHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a7& nLZ?i???#g????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a7& nLZ?i7??Z?????Unknown
?eHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a7& nLZ?i??ߑ?????Unknown
?fHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a7& nLZ?i]????????Unknown
}gHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a7& nLZ?i?????????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU