"?e
BHostIDLE"IDLE1     ??@A     ??@aX??????iX???????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      X@9      X@A      X@I      X@a|?"?P_??i?¹F?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?P@9     ?P@A     ?P@I     ?P@aڇ|???i???)-???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      N@9      N@A      N@I      N@a?]k?$7??i+?'????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?A@9     ?A@A     ?A@I     ?A@a????
??i?7l1????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      A@9      A@A      =@I      =@a A???=|?i???????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      :@9      :@A      :@I      :@a??:??Qy?i^3v?PQ???Unknown
iHostWriteSummary"WriteSummary(1      8@9      8@A      8@I      8@a|?"?P_w?i?x??????Unknown?
d	HostDataset"Iterator::Model(1     ?]@9     ?]@A      6@I      6@ax
??lu?i??,??????Unknown
g
HostStridedSlice"strided_slice(1      4@9      4@A      4@I      4@a?>??zs?i.rZ/?????Unknown
XHostEqual"Equal(1      0@9      0@A      0@I      0@aP????)o?i??K?????Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      0@9      0@A      0@I      0@aP????)o?i\y=?0???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      *@9      *@A      *@I      *@a??:??Qi?iG???)???Unknown
XHostCast"Cast_3(1      (@9      (@A      (@I      (@a|?"?P_g?i?ֶ??@???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      (@9      (@A      (@I      (@a|?"?P_g?i??k?AX???Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      (@9      (@A      (@I      (@a|?"?P_g?iZ!??o???Unknown
vHostSub"%binary_crossentropy/logistic_loss/sub(1      (@9      (@A      (@I      (@a|?"?P_g?i????????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      &@9      &@A      &@I      &@ax
??le?i?I|?l????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      $@9      $@A      $@I      $@a?>??zc?i?;??????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@aڇ|?a?i??*n????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@aڇ|?a?i??"??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aڇ|?a?i?ɪ#}????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      "@9      "@A      "@I      "@aڇ|?a?i֣2?????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aP????)_?i?e??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @aP????)_?in'$a.???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aP????)_?i:??A?$???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1       @9       @A       @I       @aP????)_?i?"X4???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @aP????)_?i?l??C???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1       @9       @A       @I       @aP????)_?i?.??S???Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @aP????)_?ij??c???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @af$SӈD[?i????p???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @af$SӈD[?i?CSL[~???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @af$SӈD[?i ????????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @af$SӈD[?i??&՟????Unknown
?#HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @af$SӈD[?iD@?B????Unknown
?$HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a|?"?P_W?i?????????Unknown
v%HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a|?"?P_W?i?bEj?????Unknown
}&HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a|?"?P_W?iO??Q????Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?>??zS?inm?????Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a?>??zS?i??6+?????Unknown
?)HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?>??zS?i?_?7?????Unknown
?*HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?>??zS?i???CE????Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?>??zS?i?QP????Unknown
o,HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a?>??zS?i	?d\????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aP????)O?i?+?̉???Unknown
X.HostCast"Cast_4(1      @9      @A      @I      @aP????)O?iՌ?<T???Unknown
\/HostGreater"Greater(1      @9      @A      @I      @aP????)O?i??????Unknown
e0Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aP????)O?i?NV?#???Unknown?
?1HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aP????)O?i?????+???Unknown
~2HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aP????)O?im??}3???Unknown
?3HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aP????)O?iSqnH;???Unknown
?4HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aP????)O?i9?G?C???Unknown
?5HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aP????)O?i3?N?J???Unknown
}6HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aP????)O?i????R???Unknown
t7HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a|?"?P_G?i????X???Unknown
V8HostCast"Cast(1      @9      @A      @I      @a|?"?P_G?i]%gW^???Unknown
X9HostCast"Cast_5(1      @9      @A      @I      @a|?"?P_G?i	nH;/d???Unknown
V:HostMean"Mean(1      @9      @A      @I      @a|?"?P_G?i??uj???Unknown
d;HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a|?"?P_G?ia????o???Unknown
r<HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a|?"?P_G?iHз?u???Unknown
b=HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a|?"?P_G?i?????{???Unknown
~>HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a|?"?P_G?ie?*`f????Unknown
?HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      @9      @A      @I      @a|?"?P_G?i"X4>????Unknown
?@HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a|?"?P_G?i?j?????Unknown
?AHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a|?"?P_G?ii????????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aP????)??i???Ӗ???Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @aP????)??iO?L?????Unknown
uDHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @aP????)??i?D??????Unknown
|EHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aP????)??i5u+??????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aP????)??i??I?g????Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aP????)??i?g-M????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aP????)??i??e2????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aP????)??i7??????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aP????)??itg???????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aP????)??i????????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @aP????)??iZ??Eǽ???Unknown
?MHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @aP????)??i??~?????Unknown
?NHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aP????)??i@);??????Unknown
?OHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @aP????)??i?YY?v????Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aP????)??i&?w&\????Unknown
}QHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aP????)??i???^A????Unknown
?RHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aP????)??i볖&????Unknown
?SHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @aP????)??i??????Unknown
?THostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @aP????)??i?K??????Unknown
?UHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @aP????)??ie|??????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aP????)/?i????????Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??aP????)/?i׬,w?????Unknown?
TXHostMul"Mul(1      ??9      ??A      ??I      ??aP????)/?i?;?????Unknown
sYHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??aP????)/?iI?J??????Unknown
uZHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??aP????)/?i??YK?????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aP????)/?i?i??????Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aP????)/?i?%x?x????Unknown
x]HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??aP????)/?i->?k????Unknown
?^HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aP????)/?ifV??]????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aP????)/?i?n?WP????Unknown
?`HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aP????)/?i؆??B????Unknown
?aHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??aP????)/?i?Ï5????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aP????)/?iJ??+(????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aP????)/?i????????Unknown
?dHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??aP????)/?i???c????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aP????)/?i?????????Unknown
?fHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aP????)/?i?N? ???Unknown
~gHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??aP????)/?i4?????Unknown
[hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i4?????Unknown*?d
sHostDataset"Iterator::Model::ParallelMapV2(1      X@9      X@A      X@I      X@a???̾?i???̾??Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?P@9     ?P@A     ?P@I     ?P@a?RKE,??i	&????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      N@9      N@A      N@I      N@a?3?τ???i?}s?????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?A@9     ?A@A     ?A@I     ?A@aLg1??t??i?I#'?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      A@9      A@A      =@I      =@a?)ѦD???i?n???????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      :@9      :@A      :@I      :@a?
?+????i[?qA????Unknown
iHostWriteSummary"WriteSummary(1      8@9      8@A      8@I      8@a???̞?i'????????Unknown?
dHostDataset"Iterator::Model(1     ?]@9     ?]@A      6@I      6@a???;??ibˍ-7????Unknown
g	HostStridedSlice"strided_slice(1      4@9      4@A      4@I      4@a???j???i?5??P???Unknown
X
HostEqual"Equal(1      0@9      0@A      0@I      0@a?H"???i??U?W????Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      0@9      0@A      0@I      0@a?H"???i;?p???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      *@9      *@A      *@I      *@a?
?+????ie?=`????Unknown
XHostCast"Cast_3(1      (@9      (@A      (@I      (@a???̎?id\?q???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      (@9      (@A      (@I      (@a???̎?i?0{?????Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      (@9      (@A      (@I      (@a???̎?i~&???g???Unknown
vHostSub"%binary_crossentropy/logistic_loss/sub(1      (@9      (@A      (@I      (@a???̎?i1.ȸ ????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      &@9      &@A      &@I      &@a???;??i@5?T???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      $@9      $@A      $@I      $@a???j???i?;????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?qA???ipA????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?qA???i6G?}s???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?qA???i?L?3?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      "@9      "@A      "@I      "@a?qA???i?RKE,???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?H"???i?W?_e~???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?H"???i]!t?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?H"???i(b???"???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1       @9       @A       @I       @a?H"???iJg1??t???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a?H"???ill???????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1       @9       @A       @I       @a?H"???i?qA????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @a?H"???i?v??%k???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ap?}???i.{??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?}???i????????Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?}???i*???B???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?}???i???"?????Unknown
?"HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ap?}???i&??4r????Unknown
?#HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a????~?i ?D
???Unknown
v$HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a????~?iڔhS?M???Unknown
}%HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a????~?i???b:????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a???j?y?i難o?????Unknown
V'HostSum"Sum_2(1      @9      @A      @I      @a???j?y?i?x|?????Unknown
?(HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???j?y?iS?M?6%???Unknown
?)HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???j?y?i??"??X???Unknown
?*HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???j?y?i????ދ???Unknown
o+HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a???j?y?i??̯2????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?H"?t?i???B????Unknown
X-HostCast"Cast_4(1      @9      @A      @I      @a?H"?t?i?T?R???Unknown
\.HostGreater"Greater(1      @9      @A      @I      @a?H"?t?i????b:???Unknown
e/Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?H"?t?i6???rc???Unknown?
?0HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?H"?t?iǸ れ???Unknown
~1HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?H"?t?iX?d풵???Unknown
?2HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?H"?t?i齨??????Unknown
?3HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?H"?t?iz??????Unknown
?4HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?H"?t?i?0?0???Unknown
}5HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?H"?t?i??t?Y???Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a????n?i??'?x???Unknown
V7HostCast"Cast(1      @9      @A      @I      @a????n?iv??%k????Unknown
X8HostCast"Cast_5(1      @9      @A      @I      @a????n?icˍ-7????Unknown
V9HostMean"Mean(1      @9      @A      @I      @a????n?iP?@5????Unknown
d:HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a????n?i=??<?????Unknown
r;HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a????n?i*ѦD????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a????n?i?YLg1???Unknown
~=HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a????n?i?T3P???Unknown
>HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      @9      @A      @I      @a????n?i?ֿ[?n???Unknown
??HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????n?i??rcˍ???Unknown
?@HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????n?i??%k?????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?H"?d?i?Gp????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?H"?d?i\?iu?????Unknown
uCHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a?H"?d?i?ދz/????Unknown
|DHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?H"?d?i?߭?????Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?H"?d?i4?τ????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?H"?d?i|????'???Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?H"?d?i???O<???Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?H"?d?i?5??P???Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?H"?d?iT?W?_e???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?H"?d?i??y??y???Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?H"?d?i?蛣o????Unknown
?LHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?H"?d?i,꽨?????Unknown
?MHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?H"?d?it?߭????Unknown
?NHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?H"?d?i???????Unknown
?OHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?H"?d?i?#??????Unknown
}PHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?H"?d?iL?E?????Unknown
?QHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?H"?d?i??g	???Unknown
?RHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a?H"?d?i????'???Unknown
?SHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a?H"?d?i$??̯2???Unknown
?THostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?H"?d?il???7G???Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?H"?T?i?^?{Q???Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??a?H"?T?i???ֿ[???Unknown?
TWHostMul"Mul(1      ??9      ??A      ??I      ??a?H"?T?iX???f???Unknown
sXHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a?H"?T?i???Gp???Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?H"?T?i???ދz???Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?H"?T?iD?3?τ???Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?H"?T?i????????Unknown
x\HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?H"?T?i??U?W????Unknown
?]HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?H"?T?i0??蛣???Unknown
?^HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?H"?T?i??w?߭???Unknown
?_HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?H"?T?ix??#????Unknown
?`HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?H"?T?i???g????Unknown
?aHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?H"?T?i??*??????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?H"?T?id????????Unknown
?cHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?H"?T?i?L?3????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?H"?T?i????w????Unknown
?eHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?H"?T?iP?n??????Unknown
~fHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a?H"?T?i?????????Unknown
[gHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown2CPU