"?R
BHostIDLE"IDLE1     ͼ@A     ͼ@a???d????i???d?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a6?$a??i\
G??t???Unknown?
uHost_FusedMatMul"sequential_19/dense_59/Relu(1     ?E@9     ?E@A     ?E@I     ?E@a(Q??l?t?i??t?O????Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a(????9h?i??ez?????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a?-/'Sac?i%???????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      3@9      3@A      .@I      .@a??ƺ?]?i?p??s????Unknown
lHostIteratorGetNext"IteratorGetNext(1      (@9      (@A      (@I      (@a?6???AW?i?Yh?????Unknown
jHostMean"binary_crossentropy/Mean(1      &@9      &@A      &@I      &@aL??uQU?i??Q?????Unknown
?	HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      $@9      $@A      $@I      $@a?-/'SaS?i????m????Unknown
?
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      "@I      "@a'??<1qQ?ig #?&???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a'??<1qQ?i<??,?	???Unknown?
^HostGatherV2"GatherV2(1       @9       @A       @I       @a)I?O?iN?j?????Unknown
dHostDataset"Iterator::Model(1      <@9      <@A       @I       @a)I?O?i`<`???Unknown
HostMatMul"+gradient_tape/sequential_19/dense_60/MatMul(1       @9       @A       @I       @a)I?O?ira?? !???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a@u??!K?i?~q:?'???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a@u??!K?i?%??.???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a@u??!K?ib??'z5???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a@u??!K?i?֍?B<???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a@u??!K?i?AC???Unknown
HostMatMul"+gradient_tape/sequential_19/dense_61/MatMul(1      @9      @A      @I      @a@u??!K?iR???I???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?6???AG?i????O???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?6???AG?in?sWtU???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?6???AG?i??2?D[???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?6???AG?i???"a???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_19/dense_59/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?6???AG?iذ??f???Unknown
?HostMatMul"-gradient_tape/sequential_19/dense_60/MatMul_1(1      @9      @A      @I      @a?6???AG?i??o??l???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_19/dense_61/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?6???AG?i4?.T?r???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      D@9      D@A      @I      @a?-/'SaC?i????^w???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?-/'SaC?i?X??6|???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?-/'SaC?i?$?R????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a)I???i??`??????Unknown
? HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a)I???i?j5?ψ???Unknown
V!HostMean"Mean(1      @9      @A      @I      @a)I???i?
?????Unknown
u"HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a)I???i???a?????Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a)I???i?S??p????Unknown
?$HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a)I???i????P????Unknown
~%HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a)I???iԙ\-1????Unknown
v&HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a)I???i?<1q????Unknown
?'HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a)I???i????????Unknown
(HostMatMul"+gradient_tape/sequential_19/dense_59/MatMul(1      @9      @A      @I      @a)I???i????ѧ???Unknown
?)HostBiasAddGrad"8gradient_tape/sequential_19/dense_60/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)I???i?%?<?????Unknown
?*HostMatMul"-gradient_tape/sequential_19/dense_61/MatMul_1(1      @9      @A      @I      @a)I???iɃ??????Unknown
?+HostReadVariableOp",sequential_19/dense_61/MatMul/ReadVariableOp(1      @9      @A      @I      @a)I???i
lX?r????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?6???A7?iQ?7?Z????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?6???A7?i?`*C????Unknown
X.HostCast"Cast_3(1      @9      @A      @I      @a?6???A7?i???\+????Unknown
X/HostEqual"Equal(1      @9      @A      @I      @a?6???A7?i&U֏????Unknown
\0HostGreater"Greater(1      @9      @A      @I      @a?6???A7?imϵ??????Unknown
?1HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?6???A7?i?I???????Unknown
b2HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?6???A7?i??t(?????Unknown
?3HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      @9      @A      @I      @a?6???A7?iB>T[?????Unknown
?4HostReluGrad"-gradient_tape/sequential_19/dense_60/ReluGrad(1      @9      @A      @I      @a?6???A7?i??3??????Unknown
?5HostReadVariableOp",sequential_19/dense_59/MatMul/ReadVariableOp(1      @9      @A      @I      @a?6???A7?i?2??????Unknown
u6Host_FusedMatMul"sequential_19/dense_60/Relu(1      @9      @A      @I      @a?6???A7?i???l????Unknown
x7Host_FusedMatMul"sequential_19/dense_61/BiasAdd(1      @9      @A      @I      @a?6???A7?i^'?&U????Unknown
s8HostSigmoid"sequential_19/dense_61/Sigmoid(1      @9      @A      @I      @a?6???A7?i???Y=????Unknown
?9HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a)I?/?i*??{-????Unknown
d:HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a)I?/?i?D??????Unknown
v;HostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a)I?/?i4?p?????Unknown
z<HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a)I?/?i??Z??????Unknown
v=HostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a)I?/?i>9E?????Unknown
v>HostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a)I?/?iÊ/%?????Unknown
`?HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a)I?/?iH?G?????Unknown
u@HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a)I?/?i?-i?????Unknown
wAHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a)I?/?iR????Unknown
?BHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a)I?/?i??ج?????Unknown
?CHostReadVariableOp"-sequential_19/dense_59/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a)I?/?i\"?Ύ????Unknown
?DHostReadVariableOp"-sequential_19/dense_61/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a)I?/?i?s??~????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a)I??i???w????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a)I??ieŗo????Unknown
XGHostCast"Cast_4(1      ??9      ??A      ??I      ??a)I??i'??#g????Unknown
aHHostIdentity"Identity(1      ??9      ??A      ??I      ??a)I??i??4_????Unknown?
TIHostMul"Mul(1      ??9      ??A      ??I      ??a)I??i??wEW????Unknown
sJHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a)I??imhlVO????Unknown
|KHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a)I??i/?agG????Unknown
rLHostAdd"!binary_crossentropy/logistic_loss(1      ??9      ??A      ??I      ??a)I??i??Vx?????Unknown
}MHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a)I??i??K?7????Unknown
wNHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a)I??iuA?/????Unknown
?OHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a)I??i746?'????Unknown
?PHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a)I??i?\+?????Unknown
?QHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a)I??i?? ?????Unknown
?RHostReluGrad"-gradient_tape/sequential_19/dense_59/ReluGrad(1      ??9      ??A      ??I      ??a)I??i}??????Unknown
?SHostReadVariableOp"-sequential_19/dense_60/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a)I??i??
?????Unknown
?THostReadVariableOp",sequential_19/dense_60/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a)I??i     ???Unknown
LUHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown*?Q
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a4M?x???i4M?x????Unknown?
uHost_FusedMatMul"sequential_19/dense_59/Relu(1     ?E@9     ?E@A     ?E@I     ?E@a0?*??]??i????1???Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a?Tsሮ??i???9l????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a	?????i???r?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      3@9      3@A      .@I      .@a?#?
k??i?:???????Unknown
lHostIteratorGetNext"IteratorGetNext(1      (@9      (@A      (@I      (@aq2?????i?GY??S???Unknown
jHostMean"binary_crossentropy/Mean(1      &@9      &@A      &@I      &@a=???ׄ?izӛ??????Unknown
?HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      $@9      $@A      $@I      $@a	?????i"???????Unknown
?	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      "@I      "@a?e?d??i?g??6???Unknown
e
Host
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a?e?d??iP?-.%{???Unknown?
^HostGatherV2"GatherV2(1       @9       @A       @I       @aBC^P~?i???Dŷ???Unknown
dHostDataset"Iterator::Model(1      <@9      <@A       @I       @aBC^P~?i^?[e????Unknown
HostMatMul"+gradient_tape/sequential_19/dense_60/MatMul(1       @9       @A       @I       @aBC^P~?i?
br1???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aٺC?	?z?i[?F?f???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @aٺC?	?z?i?+?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aٺC?	?z?iG??)????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aٺC?	?z?i?(??5???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aٺC?	?z?i3???A:???Unknown
HostMatMul"+gradient_tape/sequential_19/dense_61/MatMul(1      @9      @A      @I      @aٺC?	?z?i?7??Mo???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aq2???v?i>??Ŝ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aq2???v?isD?>????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aq2???v?i?J??????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aq2???v?i=Q?-.%???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_19/dense_59/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aq2???v?i?W?>?R???Unknown
?HostMatMul"-gradient_tape/sequential_19/dense_60/MatMul_1(1      @9      @A      @I      @aq2???v?i^P????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_19/dense_61/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aq2???v?ilda?????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      D@9      D@A      @I      @a	???r?i??Moz????Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a	???r?io?}^????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a	???r?ih???B???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aBC^Pn?i????=???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aBC^Pn?i??t??[???Unknown
V HostMean"Mean(1      @9      @A      @I      @aBC^Pn?i1ӭ2z???Unknown
u!HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aBC^Pn?it1??????Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @aBC^Pn?i?	??Ҷ???Unknown
?#HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aBC^Pn?i???"????Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aBC^Pn?i=K?r????Unknown
v%HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aBC^Pn?i???????Unknown
?&HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aBC^Pn?i??0???Unknown
'HostMatMul"+gradient_tape/sequential_19/dense_59/MatMul(1      @9      @A      @I      @aBC^Pn?ie?bN???Unknown
?(HostBiasAddGrad"8gradient_tape/sequential_19/dense_60/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aBC^Pn?iI#??l???Unknown
?)HostMatMul"-gradient_tape/sequential_19/dense_61/MatMul_1(1      @9      @A      @I      @aBC^Pn?i?'!????Unknown
?*HostReadVariableOp",sequential_19/dense_61/MatMul/ReadVariableOp(1      @9      @A      @I      @aBC^Pn?i?+S????Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aq2???f?i?(????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aq2???f?i32?0?????Unknown
X-HostCast"Cast_3(1      @9      @A      @I      @aq2???f?ie?9?????Unknown
X.HostEqual"Equal(1      @9      @A      @I      @aq2???f?i?8?AC???Unknown
\/HostGreater"Greater(1      @9      @A      @I      @aq2???f?iɻJ????Unknown
?0HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aq2???f?i?>?R?1???Unknown
b1HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aq2???f?i-?,[wH???Unknown
?2HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      @9      @A      @I      @aq2???f?i_E?c3_???Unknown
?3HostReluGrad"-gradient_tape/sequential_19/dense_60/ReluGrad(1      @9      @A      @I      @aq2???f?i??9l?u???Unknown
?4HostReadVariableOp",sequential_19/dense_59/MatMul/ReadVariableOp(1      @9      @A      @I      @aq2???f?i?K?t?????Unknown
u5Host_FusedMatMul"sequential_19/dense_60/Relu(1      @9      @A      @I      @aq2???f?i??F}g????Unknown
x6Host_FusedMatMul"sequential_19/dense_61/BiasAdd(1      @9      @A      @I      @aq2???f?i'Rͅ#????Unknown
s7HostSigmoid"sequential_19/dense_61/Sigmoid(1      @9      @A      @I      @aq2???f?iY?S??????Unknown
?8HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @aBC^P^?i{??????Unknown
d9HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aBC^P^?i?ٱ?/????Unknown
v:HostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aBC^P^?i??`?W????Unknown
z;HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @aBC^P^?i??????Unknown
v<HostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @aBC^P^?iྪ????Unknown
v=HostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aBC^P^?i%?m??+???Unknown
`>HostDivNoNan"
div_no_nan(1       @9       @A       @I       @aBC^P^?iG???:???Unknown
u?HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aBC^P^?ii?˻J???Unknown
w@HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aBC^P^?i??z?GY???Unknown
?AHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @aBC^P^?i??)?oh???Unknown
?BHostReadVariableOp"-sequential_19/dense_59/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aBC^P^?i???̗w???Unknown
?CHostReadVariableOp"-sequential_19/dense_61/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aBC^P^?i???ҿ????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aBC^PN?ip_?S????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??aBC^PN?i?6??????Unknown
XFHostCast"Cast_4(1      ??9      ??A      ??I      ??aBC^PN?i$r?{????Unknown
aGHostIdentity"Identity(1      ??9      ??A      ??I      ??aBC^PN?i5???????Unknown?
THHostMul"Mul(1      ??9      ??A      ??I      ??aBC^PN?iFt?࣬???Unknown
sIHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??aBC^PN?iW???7????Unknown
|JHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aBC^PN?ihvl?˻???Unknown
rKHostAdd"!binary_crossentropy/logistic_loss(1      ??9      ??A      ??I      ??aBC^PN?iy?C?_????Unknown
}LHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aBC^PN?i?x??????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aBC^PN?i?????????Unknown
?NHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aBC^PN?i?z??????Unknown
?OHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aBC^PN?i?????????Unknown
?PHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??aBC^PN?i?|y?C????Unknown
?QHostReluGrad"-gradient_tape/sequential_19/dense_59/ReluGrad(1      ??9      ??A      ??I      ??aBC^PN?i??P??????Unknown
?RHostReadVariableOp"-sequential_19/dense_60/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aBC^PN?i?~(?k????Unknown
?SHostReadVariableOp",sequential_19/dense_60/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aBC^PN?i      ???Unknown
LTHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i      ???Unknown2CPU