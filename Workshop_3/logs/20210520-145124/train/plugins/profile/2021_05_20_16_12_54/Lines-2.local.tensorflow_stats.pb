"?g
BHostIDLE"IDLE1    ??@A    ??@aI??O??iI??O???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     `?@9     `?@A     `?@I     `?@a?8|0+h??i[&%UX???Unknown?
iHostWriteSummary"WriteSummary(1      D@9      D@A      D@I      D@asSs??sp?i1"=y???Unknown?
wHost_FusedMatMul"sequential_8/dense_26/BiasAdd(1      @@9      @@A      @@I      @@aQR??0Sj?iT?S?????Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      9@I      9@aO(P3??d?i|?IQ!????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      4@9      4@A      3@I      3@a?Q??B_?i?)R?·???Unknown
lHostTanh"sequential_8/dense_26/Tanh(1      0@9      0@A      0@I      0@aQR??0SZ?iN??N?????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      ,@9      ,@A      ,@I      ,@a????W?i?V)?p????Unknown
r	HostSigmoid"sequential_8/dense_28/Sigmoid(1      *@9      *@A      *@I      *@a?R|??cU?i˔?"????Unknown
V
HostCast"Cast(1      &@9      &@A      &@I      &@a??˝1R?i?z?/????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@asSs??sP?iI4i????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      $@9      $@A      $@I      $@asSs??sP?i??d?????Unknown
wHost_FusedMatMul"sequential_8/dense_27/BiasAdd(1      $@9      $@A      $@I      $@asSs??sP?i????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      .@9      .@A      "@I      "@a??5??M?i5m|D???Unknown
?HostMatMul",gradient_tape/sequential_8/dense_27/MatMul_1(1      "@9      "@A      "@I      "@a??5??M?i??-?????Unknown
~HostMatMul"*gradient_tape/sequential_8/dense_28/MatMul(1      "@9      "@A      "@I      "@a??5??M?iP?G???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aQR??0SJ?io?'????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aQR??0SJ?i?a?< ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @aQR??0SJ?it???&???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @aQR??0SJ?in??xf-???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @aQR??0SJ?i?6E?3???Unknown
~HostMatMul"*gradient_tape/sequential_8/dense_27/MatMul(1       @9       @A       @I       @aQR??0SJ?i?F?:???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @aQR??0SJ?im??$A???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a????G?i?.2?F???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a????G?i?c?B?L???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a????G?i똖ukR???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a????G?i?H?-X???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??#?d?C?i?sA]???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??#?d?C?i???b???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??#?d?C?i??s?f???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??#?d?C?i???k???Unknown
? HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??#?d?C?i???p???Unknown
~!HostMatMul"*gradient_tape/sequential_8/dense_26/MatMul(1      @9      @A      @I      @a??#?d?C?iK??u???Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @asSs??s@?i???>?y???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @asSs??s@?i???>~???Unknown
V$HostSum"Sum_2(1      @9      @A      @I      @asSs??s@?i??6>"????Unknown
v%HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @asSs??s@?icw?=?????Unknown
?&HostBiasAddGrad"7gradient_tape/sequential_8/dense_27/BiasAdd/BiasAddGrad(1      @9      @A      @I      @asSs??s@?i8T~=\????Unknown
?'HostBiasAddGrad"7gradient_tape/sequential_8/dense_28/BiasAdd/BiasAddGrad(1      @9      @A      @I      @asSs??s@?i1"=y????Unknown
u(HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aQR??0S:?i??>?Ñ???Unknown
?)HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aQR??0S:?ia?[	????Unknown
?*HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aQR??0S:?iCxoX????Unknown
?+HostBiasAddGrad"7gradient_tape/sequential_8/dense_26/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aQR??0S:?i???բ????Unknown
?,HostMatMul",gradient_tape/sequential_8/dense_28/MatMul_1(1      @9      @A      @I      @aQR??0S:?i_??;?????Unknown
?-HostReadVariableOp"+sequential_8/dense_27/MatMul/ReadVariableOp(1      @9      @A      @I      @aQR??0S:?i	UΡ7????Unknown
w.Host_FusedMatMul"sequential_8/dense_28/BiasAdd(1      @9      @A      @I      @aQR??0S:?i???????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??#?d?3?i3????????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??#?d?3?i??q????Unknown
X1HostCast"Cast_3(1      @9      @A      @I      @a??#?d?3?i3??m?????Unknown
`2HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??#?d?3?i?A:a????Unknown
\3HostGreater"Greater(1      @9      @A      @I      @a??#?d?3?i3??ٱ???Unknown
?4HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      @I      @a??#?d?3?i? l?P????Unknown
V5HostMean"Mean(1      @9      @A      @I      @a??#?d?3?i3??ȶ???Unknown
s6HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??#?d?3?i?)?l@????Unknown
d7HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a??#?d?3?i3?,9?????Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??#?d?3?i?2?0????Unknown
v9HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??#?d?3?i3?Wҧ????Unknown
b:HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??#?d?3?i?;??????Unknown
?;Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a??#?d?3?i3??k?????Unknown
?<HostReadVariableOp"+sequential_8/dense_26/MatMul/ReadVariableOp(1      @9      @A      @I      @a??#?d?3?i?D8????Unknown
?=HostReadVariableOp",sequential_8/dense_27/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??#?d?3?i3ɭ?????Unknown
?>HostReadVariableOp",sequential_8/dense_28/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??#?d?3?i?MC??????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aQR??0S*?i?Q?????Unknown
X@HostCast"Cast_5(1       @9       @A       @I       @aQR??0S*?i]?_7I????Unknown
XAHostEqual"Equal(1       @9       @A       @I       @aQR??0S*?i?Vnj?????Unknown
TBHostMul"Mul(1       @9       @A       @I       @aQR??0S*?i?|??????Unknown
|CHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aQR??0S*?i\??8????Unknown
jDHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aQR??0S*?i?_??????Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aQR??0S*?i??6?????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aQR??0S*?i[?i(????Unknown
zGHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @aQR??0S*?i?hĜ?????Unknown
vHHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aQR??0S*?i???r????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aQR??0S*?iZ?????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aQR??0S*?i?q?5?????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aQR??0S*?i??hb????Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aQR??0S*?iY"?????Unknown
?MHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @aQR??0S*?i?zϬ????Unknown
~NHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @aQR??0S*?i?(R????Unknown
?OHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aQR??0S*?iX+75?????Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aQR??0S*?i??Eh?????Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aQR??0S*?i?S?A????Unknown
?RHostTanhGrad",gradient_tape/sequential_8/dense_26/TanhGrad(1       @9       @A       @I       @aQR??0S*?iW4b??????Unknown
?SHostReadVariableOp",sequential_8/dense_26/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aQR??0S*?i??p?????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aQR??0S?i׸??^????Unknown
aUHostIdentity"Identity(1      ??9      ??A      ??I      ??aQR??0S?i?~41????Unknown?
?VHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??aQR??0S?i-?????Unknown
}WHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aQR??0S?iX=?g?????Unknown
wXHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aQR??0S?i?i?????Unknown
xYHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??aQR??0S?i????{????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aQR??0S?i??"4N????Unknown
?[HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aQR??0S?i??? ????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aQR??0S?i/1g?????Unknown
?]HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aQR??0S?iZF? ?????Unknown
?^HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??aQR??0S?i?r???????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aQR??0S?i???3k????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aQR??0S?i??M?=????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aQR??0S?i??f????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aQR??0S?i1#\ ?????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aQR??0S?i\O㙵????Unknown
~dHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??aQR??0S?i?{j3?????Unknown
?eHostTanhGrad",gradient_tape/sequential_8/dense_27/TanhGrad(1      ??9      ??A      ??I      ??aQR??0S?i????Z????Unknown
lfHostTanh"sequential_8/dense_27/Tanh(1      ??9      ??A      ??I      ??aQR??0S?i??xf-????Unknown
?gHostReadVariableOp"+sequential_8/dense_28/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aQR??0S?i     ???Unknown
+hHostCast"Cast_4(i     ???Unknown
LiHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
YjHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown*?f
uHostFlushSummaryWriter"FlushSummaryWriter(1     `?@9     `?@A     `?@I     `?@a^b)w͜??i^b)w͜???Unknown?
iHostWriteSummary"WriteSummary(1      D@9      D@A      D@I      D@a??#?
???iY?? ?????Unknown?
wHost_FusedMatMul"sequential_8/dense_26/BiasAdd(1      @@9      @@A      @@I      @@aRL?^w???i?????{???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      9@I      9@a?s?A?˔?iY??EP"???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      4@9      4@A      3@I      3@a?*]?-???in???????Unknown
lHostTanh"sequential_8/dense_26/Tanh(1      0@9      0@A      0@I      0@aRL?^w???i5?[?:???Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      ,@9      ,@A      ,@I      ,@a?b?r?J??i?'|eh???Unknown
rHostSigmoid"sequential_8/dense_28/Sigmoid(1      *@9      *@A      *@I      *@an??????ix???????Unknown
V	HostCast"Cast(1      &@9      &@A      &@I      &@ay??L??i??^H???Unknown
l
HostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a??#?
???i?b?r?J???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      $@9      $@A      $@I      $@a??#?
???i?7?4????Unknown
wHost_FusedMatMul"sequential_8/dense_27/BiasAdd(1      $@9      $@A      $@I      $@a??#?
???iG???????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      .@9      .@A      "@I      "@a?5sJF?}?i?e9T????Unknown
?HostMatMul",gradient_tape/sequential_8/dense_27/MatMul_1(1      "@9      "@A      "@I      "@a?5sJF?}?iL???G???Unknown
~HostMatMul"*gradient_tape/sequential_8/dense_28/MatMul(1      "@9      "@A      "@I      "@a?5sJF?}?i?2cmn????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aRL?^w?z?i$q \?????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aRL?^w?z?i???J?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @aRL?^w?z?iV??9%#???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @aRL?^w?z?i?,X(bX???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @aRL?^w?z?i?k?????Unknown
~HostMatMul"*gradient_tape/sequential_8/dense_27/MatMul(1       @9       @A       @I       @aRL?^w?z?i!???????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @aRL?^w?z?i????????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?b?r?Jw?i?uE?&???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?b?r?Jw?iF[?CU???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?b?r?Jw?i?@?؃???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?b?r?Jw?i?C&8n????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a>y????s?i?24?[????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a>y????s?i?!B?I???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a>y????s?i?PQ7*???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a>y????s?i??]%R???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a>y????s?i??k?z???Unknown
~ HostMatMul"*gradient_tape/sequential_8/dense_26/MatMul(1      @9      @A      @I      @a>y????s?i~?yj ????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??#?
?p?i?$?F????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??#?
?p?i?k攌????Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a??#?
?p?i۲?????Unknown
v$HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??#?
?p?i??R?'???Unknown
?%HostBiasAddGrad"7gradient_tape/sequential_8/dense_27/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??#?
?p?iA??^H???Unknown
?&HostBiasAddGrad"7gradient_tape/sequential_8/dense_28/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??#?
?p?i8????i???Unknown
u'HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aRL?^w?j?i?'aC????Unknown
?(HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aRL?^w?j?i??|??????Unknown
?)HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aRL?^w?j?if?O?????Unknown
?*HostBiasAddGrad"7gradient_tape/sequential_8/dense_26/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aRL?^w?j?ih:?????Unknown
?+HostMatMul",gradient_tape/sequential_8/dense_28/MatMul_1(1      @9      @A      @I      @aRL?^w?j?i???>?????Unknown
?,HostReadVariableOp"+sequential_8/dense_27/MatMul/ReadVariableOp(1      @9      @A      @I      @aRL?^w?j?i D??[	???Unknown
w-Host_FusedMatMul"sequential_8/dense_28/BiasAdd(1      @9      @A      @I      @aRL?^w?j?iL?U-?#???Unknown
t.HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a>y????c?i????7???Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a>y????c?i>?c??K???Unknown
X0HostCast"Cast_3(1      @9      @A      @I      @a>y????c?i?????_???Unknown
`1HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a>y????c?i0?q??s???Unknown
\2HostGreater"Greater(1      @9      @A      @I      @a>y????c?i???l̇???Unknown
?3HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      @I      @a>y????c?i"?FÛ???Unknown
V4HostMean"Mean(1      @9      @A      @I      @a>y????c?i?? ?????Unknown
s5HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a>y????c?i????????Unknown
d6HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a>y????c?i??ӧ????Unknown
?7HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a>y????c?i????????Unknown
v8HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a>y????c?i?"??????Unknown
b9HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a>y????c?i?|?_????Unknown
?:Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a>y????c?iqt09?'???Unknown
?;HostReadVariableOp"+sequential_8/dense_26/MatMul/ReadVariableOp(1      @9      @A      @I      @a>y????c?i?k?z;???Unknown
?<HostReadVariableOp",sequential_8/dense_27/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a>y????c?icc>?pO???Unknown
?=HostReadVariableOp",sequential_8/dense_28/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a>y????c?i?Z??gc???Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aRL?^w?Z?i??t?p???Unknown
X?HostCast"Cast_5(1       @9       @A       @I       @aRL?^w?Z?i(?#=~???Unknown
X@HostEqual"Equal(1       @9       @A       @I       @aRL?^w?Z?i?I?xU????Unknown
TAHostMul"Mul(1       @9       @A       @I       @aRL?^w?Z?it????????Unknown
|BHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aRL?^w?Z?i?1??????Unknown
jCHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aRL?^w?Z?i?8?+C????Unknown
rDHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aRL?^w?Z?if??g?????Unknown
vEHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aRL?^w?Z?i????????Unknown
zFHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @aRL?^w?Z?i?'??0????Unknown
vGHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aRL?^w?Z?iXw??????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aRL?^w?Z?i??MV?????Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aRL?^w?Z?i??????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aRL?^w?Z?iJf??m???Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aRL?^w?Z?i??[	????Unknown
?LHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @aRL?^w?Z?i?E+???Unknown
~MHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @aRL?^w?Z?i<U??[8???Unknown
?NHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aRL?^w?Z?i??i??E???Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aRL?^w?Z?i????R???Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aRL?^w?Z?i.D?3I`???Unknown
?QHostTanhGrad",gradient_tape/sequential_8/dense_26/TanhGrad(1       @9       @A       @I       @aRL?^w?Z?iԓwo?m???Unknown
?RHostReadVariableOp",sequential_8/dense_26/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aRL?^w?Z?iz?&??z???Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aRL?^w?J?iM??H?????Unknown
aTHostIdentity"Identity(1      ??9      ??A      ??I      ??aRL?^w?J?i 3??6????Unknown?
?UHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??aRL?^w?J?i?ڭ?ގ???Unknown
}VHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aRL?^w?J?iƂ?"?????Unknown
wWHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aRL?^w?J?i?*]?-????Unknown
xXHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??aRL?^w?J?il?4^բ???Unknown
?YHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aRL?^w?J?i?z?|????Unknown
?ZHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aRL?^w?J?i"??$????Unknown
?[Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aRL?^w?J?i?ɻ7̶???Unknown
?\HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aRL?^w?J?i?q??s????Unknown
?]HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??aRL?^w?J?i?ks????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aRL?^w?J?i^?B?????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aRL?^w?J?i1i?j????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aRL?^w?J?i?L????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aRL?^w?J?i׸???????Unknown
?bHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aRL?^w?J?i?`??a????Unknown
~cHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??aRL?^w?J?i}y&	????Unknown
?dHostTanhGrad",gradient_tape/sequential_8/dense_27/TanhGrad(1      ??9      ??A      ??I      ??aRL?^w?J?iP?Pİ????Unknown
leHostTanh"sequential_8/dense_27/Tanh(1      ??9      ??A      ??I      ??aRL?^w?J?i#X(bX????Unknown
?fHostReadVariableOp"+sequential_8/dense_28/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aRL?^w?J?i?????????Unknown
+gHostCast"Cast_4(i?????????Unknown
LhHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
YiHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU