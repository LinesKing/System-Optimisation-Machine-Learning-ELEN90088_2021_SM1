"?n
BHostIDLE"IDLE1     :?@A     :?@a[KO?~??i[KO?~???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a&a&a??iMd?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      Q@9      Q@A      Q@I      Q@ax?E?ڑ?i??:m????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      >@9      >@A      >@I      >@a ?????i??E*q????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ;@9      ;@A      ;@I      ;@a???O_[|?i????'???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      :@9      :@A      :@I      :@aO贁N{?i??N??N???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      6@9      6@A      6@I      6@a~>NJw?ija??|???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      2@9      2@A      2@I      2@a?a?ߔ?r?i-ʢ,ʢ???Unknown
?	Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      0@9      0@A      0@I      0@aDsg???p?i???e????Unknown
?
Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      .@9      .@A      .@I      .@a ????o?i??????Unknown
dHostDataset"Iterator::Model(1      U@9      U@A      ,@I      ,@a?	5?<hm?iPP???Unknown
\HostGreater"Greater(1      &@9      &@A      &@I      &@a~>NJg?iT?K k???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a~>NJg?i???+?/???Unknown?
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?a?ߔ?b?i??u?mB???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      "@9      "@A      "@I      "@a?a?ߔ?b?iVUUUUU???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a?a?ߔ?b?i?	5?<h???Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      "@9      "@A      "@I      "@a?a?ߔ?b?i?${???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aDsg???`?i?%?X?????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @aDsg???`?i ?i2?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @aDsg???`?is??????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @aDsg???`?i?[??[????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @aDsg???`?iY?h?)????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @aDsg???`?i?*??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?	5?<h]?iQE???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a?	5?<h]?i?_??_????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a?	5?<h]?i[zr????Unknown
oHostSigmoid"sequential/dense_3/Sigmoid(1      @9      @A      @I      @a?	5?<h]?i???????Unknown
VHostMean"Mean(1      @9      @A      @I      @a?,??4Y?ivb'vb'???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?,??4Y?i0g??3???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?,??4Y?i???<?@???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a?,??4Y?i8???1M???Unknown
} HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      @9      @A      @I      @a?,??4Y?iΘ&?Y???Unknown
a!HostCast"sequential/Cast(1      @9      @A      @I      @a?,??4Y?idfffff???Unknown
e"Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aPPU?i?p?p???Unknown?
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aPPU?i?g{?g{???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aPPU?i\??^?????Unknown
V%HostSum"Sum_2(1      @9      @A      @I      @aPPU?ii?i????Unknown
j&HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aPPU?i?隮?????Unknown
~'HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aPPU?iTj?Vj????Unknown
?(HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aPPU?i?????????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aDsg???P?i???Q????Unknown
s*HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aDsg???P?ipRZظ????Unknown
?+HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aDsg???P?i*?/?????Unknown
j,HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @aDsg???P?i????????Unknown
v-HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aDsg???P?i??ٞ?????Unknown
v.HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aDsg???P?iX!??T????Unknown
?/HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aDsg???P?iU?x?????Unknown
?0HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aDsg???P?ïYe"????Unknown
?1HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aDsg???P?i??.R?????Unknown
?2HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aDsg???P?i@??????Unknown
}3HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aDsg???P?i?#?+W???Unknown
?4HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @aDsg???P?i?W?????Unknown
q5Host_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @aDsg???P?in??%???Unknown
t6Host_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @aDsg???P?i(?X??%???Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?,??4I?i???#?+???Unknown
X8HostCast"Cast_3(1      @9      @A      @I      @a?,??4I?i???U&2???Unknown
u9HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?,??4I?i?s8?s8???Unknown
?:HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?,??4I?iTZظ?>???Unknown
?;HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?,??4I?iAx?E???Unknown
z<HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?,??4I?i?'[K???Unknown
b=HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?,??4I?i??M?Q???Unknown
?>HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?,??4I?i??W?W???Unknown
??HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?,??4I?iK???B^???Unknown
?@HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a?,??4I?i×??d???Unknown
tAHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aDsg???@?i?\Y?h???Unknown
VBHostCast"Cast(1       @9       @A       @I       @aDsg???@?i??l??l???Unknown
XCHostEqual"Equal(1       @9       @A       @I       @aDsg???@?i???E*q???Unknown
?DHostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1       @9       @A       @I       @aDsg???@?i?*B?]u???Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aDsg???@?igĬ2?y???Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aDsg???@?iD^??}???Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aDsg???@?i!???????Unknown
vHHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aDsg???@?i????+????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aDsg???@?i?+W_????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aDsg???@?i?????????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aDsg???@?i?_,?Œ???Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aDsg???@?ir??o?????Unknown
wMHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aDsg???@?iO??,????Unknown
xNHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aDsg???@?i,-l\`????Unknown
~OHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @aDsg???@?i	??ғ????Unknown
?PHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @aDsg???@?i?`AIǧ???Unknown
?QHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1       @9       @A       @I       @aDsg???@?i?????????Unknown
?RHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @aDsg???@?i??6.????Unknown
?SHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aDsg???@?i}.??a????Unknown
?THostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aDsg???@?iZ??"?????Unknown
?UHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aDsg???@?i7bV?ȼ???Unknown
?VHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @aDsg???@?i???????Unknown
~WHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aDsg???@?i??+?/????Unknown
?XHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aDsg???@?i?/??b????Unknown
}YHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aDsg???@?i?? s?????Unknown
ZHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @aDsg???@?i?ck??????Unknown
[HostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1       @9       @A       @I       @aDsg???@?ie??_?????Unknown
?\HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aDsg???@?iB?@?0????Unknown
?]HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aDsg???@?i1?Ld????Unknown
?^HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @aDsg???@?i??×????Unknown
v_HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aDsg???0?i?K~?????Unknown
v`HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??aDsg???0?i?d?9?????Unknown
XaHostCast"Cast_5(1      ??9      ??A      ??I      ??aDsg???0?iƱ???????Unknown
abHostIdentity"Identity(1      ??9      ??A      ??I      ??aDsg???0?i?????????Unknown?
TcHostMul"Mul(1      ??9      ??A      ??I      ??aDsg???0?i?K k????Unknown
|dHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aDsg???0?i??U&2????Unknown
weHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aDsg???0?i~???K????Unknown
yfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aDsg???0?il2??e????Unknown
?gHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aDsg???0?iZ?W????Unknown
?hHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aDsg???0?iH?*?????Unknown
?iHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aDsg???0?i6`β????Unknown
?jHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aDsg???0?i$f???????Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aDsg???0?i??D?????Unknown
?lHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aDsg???0?i      ???Unknown
?mHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aDsg???0?iw??????Unknown
?nHostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aDsg???0?i?L5????Unknown
+oHostCast"Cast_4(i?L5????Unknown
ipHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?L5????Unknown
YqHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?L5????Unknown
[rHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?L5????Unknown*?m
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Q@9     ?Q@A     ?Q@I     ?Q@az_兠??iz_兠???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      Q@9      Q@A      Q@I      Q@a<"??ݹ?i???$????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      >@9      >@A      >@I      >@aD-??Ҧ?i ,??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ;@9      ;@A      ;@I      ;@a?u?u???iS?t?8????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      :@9      :@A      :@I      :@aL8??ǣ?i???*/???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      6@9      6@A      6@I      6@aTC﫼??iX;??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      2@9      2@A      2@I      2@a???Gc??i?.'?????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      0@9      0@A      0@I      0@a????X??i܂?%}W???Unknown
?	Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      .@9      .@A      .@I      .@aD-??Җ?iW;??????Unknown
d
HostDataset"Iterator::Model(1      U@9      U@A      ,@I      ,@aȲ?7M??i???b|???Unknown
\HostGreater"Greater(1      &@9      &@A      &@I      &@aTC﫼??i߸?!G%???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@aTC﫼??i ???1???Unknown?
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a???Gc??i??????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      "@9      "@A      "@I      "@a???Gc??i?\?/?s???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a???Gc??i??O0????Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      "@9      "@A      "@I      "@a???Gc??i?B?n?N???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a????X??i??On????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a????X??i???m???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a????X??i??m?r???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a????X??i??+mA????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a????X??i?]?l?5???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a????X??i??hl????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aȲ?7M??i??L8????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @aȲ?7M??iВ?+mA???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @aȲ?7M??i?]i?????Unknown
oHostSigmoid"sequential/dense_3/Sigmoid(1      @9      @A      @I      @aȲ?7M??i)??????Unknown
VHostMean"Mean(1      @9      @A      @I      @aн?/B??iP ˪?4???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aн?/B??i??j?}???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aн?/B??i?9*?????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @aн?/B??i??????Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      @9      @A      @I      @aн?/B??iP???Y???Unknown
a HostCast"sequential/Cast(1      @9      @A      @I      @aн?/B??i??]i????Unknown
e!Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a???On~?i?!	?????Unknown?
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???On~?iP;??????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???On~?i?^?H?X???Unknown
V$HostSum"Sum_2(1      @9      @A      @I      @a???On~?i?j?}????Unknown
j%HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a???On~?ip?-?Z????Unknown
~&HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a???On~?i???'7???Unknown
?'HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???On~?i0???L???Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????Xx?i?;?G?|???Unknown
s)HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a????Xx?i0?R?t????Unknown
?*HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a????Xx?i??!G%????Unknown
j+HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a????Xx?i0*??????Unknown
v,HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a????Xx?i?y?F?????Unknown
v-HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a????Xx?i0ɏ?6p???Unknown
?.HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a????Xx?i?_F?????Unknown
?/HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a????Xx?i0h.Ɨ????Unknown
?0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????Xx?i???EH???Unknown
?1HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????Xx?i0???2???Unknown
}2HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a????Xx?i?V?E?c???Unknown
?3HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a????Xx?i0?k?Y????Unknown
q4Host_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a????Xx?i??:E
????Unknown
t5Host_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @a????Xx?i0E
ź????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aн?/Br?i???$????Unknown
X7HostCast"Cast_3(1      @9      @A      @I      @aн?/Br?ip<???>???Unknown
u8HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aн?/Br?i???Gc???Unknown
?9HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aн?/Br?i?3xḊ???Unknown
?:HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aн?/Br?iP?S?P????Unknown
z;HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aн?/Br?i?*/?????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aн?/Br?i??
dY????Unknown
?=HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aн?/Br?i0"??????Unknown
?>HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aн?/Br?iН?#b>???Unknown
??HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @aн?/Br?ip???b???Unknown
t@HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a????Xh?i0???>{???Unknown
VAHostCast"Cast(1       @9       @A       @I       @a????Xh?i?hl?????Unknown
XBHostEqual"Equal(1       @9       @A       @I       @a????Xh?i?TC?????Unknown
?CHostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1       @9       @A       @I       @a????Xh?ip?;?G????Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a????Xh?i0`#ß????Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a????Xh?i??????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a????Xh?i???BP???Unknown
vGHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a????Xh?ipWڂ?%???Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a????Xh?i0??? >???Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a????Xh?i???YV???Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a????Xh?i?N?B?n???Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a????Xh?ip?x?	????Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a????Xh?i0?`?a????Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a????Xh?i?EH?????Unknown
~NHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a????Xh?i??/B????Unknown
?OHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a????Xh?ip??j????Unknown
?PHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1       @9       @A       @I       @a????Xh?i0=??? ???Unknown
?QHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a????Xh?i??????Unknown
?RHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a????Xh?i???As1???Unknown
?SHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a????Xh?ip4???I???Unknown
?THostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a????Xh?i0ܝ?#b???Unknown
?UHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a????Xh?i???|z???Unknown
~VHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a????Xh?i?+mAԒ???Unknown
?WHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a????Xh?ip?T?,????Unknown
}XHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a????Xh?i0{<??????Unknown
YHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a????Xh?i?"$?????Unknown
ZHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1       @9       @A       @I       @a????Xh?i??A5????Unknown
?[HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????Xh?ipr??????Unknown
?\HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????Xh?i0???$???Unknown
?]HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a????Xh?i??? >=???Unknown
v^HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a????XX?iЕ? jI???Unknown
v_HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a????XX?i?i?@?U???Unknown
X`HostCast"Cast_5(1      ??9      ??A      ??I      ??a????XX?i?=?`?a???Unknown
aaHostIdentity"Identity(1      ??9      ??A      ??I      ??a????XX?ip???m???Unknown?
TbHostMul"Mul(1      ??9      ??A      ??I      ??a????XX?iP兠z???Unknown
|cHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a????XX?i0?y?F????Unknown
wdHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????XX?i?m?r????Unknown
yeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????XX?i?`a ?????Unknown
?fHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a????XX?i?4U ˪???Unknown
?gHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a????XX?i?I@?????Unknown
?hHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a????XX?i??<`#????Unknown
?iHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a????XX?ip?0?O????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a????XX?iP?$?{????Unknown
?kHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????XX?i0X??????Unknown
?lHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a????XX?i,??????Unknown
?mHostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a????XX?i?????????Unknown
+nHostCast"Cast_4(i?????????Unknown
ioHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
YpHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
[qHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU