"?f
BHostIDLE"IDLE1     
?@A     
?@a?/?ן$??i?/?ן$???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     @R@9     @R@A     @R@I     @R@a??I?????i}]??????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?S@9     ?S@A     ?Q@I     ?Q@a?֛)??i?[?Az???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      A@9      A@A      A@I      A@a?=h*??i????????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      >@9      >@A      :@I      :@a?pP?r?~?i?Xe?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      4@9      4@A      4@I      4@aM/*?k?w?iB??t8???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      2@9      2@A      2@I      2@a,?rZu?iʒ???b???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      0@9      0@A      0@I      0@a
Y?u??r?i|	??????Unknown
^	HostGatherV2"GatherV2(1      .@9      .@A      .@I      .@az?_??q?i???6L????Unknown
`
HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@az?_??q?i
?9X?????Unknown
VHostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a???d?p?i??"????Unknown
dHostDataset"Iterator::Model(1     @U@9     @U@A      (@I      (@a???xl?i?(?<????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      (@9      (@A      (@I      (@a???xl?i??hW	*???Unknown
oHostSigmoid"sequential/dense_2/Sigmoid(1      (@9      (@A      (@I      (@a???xl?i?Zr?F???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@an??!?j?i?<;5?`???Unknown?
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aM/*?k?g?i?fΠSx???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      $@9      $@A      $@I      $@aM/*?k?g?i??a????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@a,?rZe?i?f g????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a
Y?u??b?i???a????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a
Y?u??b?ifzQ?\????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???d?`?iT~8??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???d?`?iB?c?????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a???d?`?i0??.????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      ]@9      ]@A      @I      @a???x\?i??^?j???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???x\?i6??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a???x\?i?k??'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a???x\?i<?g?6???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???x\?i??
[D???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a???x\?iBQ?R???Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aM/*?k?W?iZ???s^???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aM/*?k?W?ir{??Pj???Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aM/*?k?W?i?u9-v???Unknown
?!HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aM/*?k?W?i??>?	????Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a
Y?u??R?iO?yM?????Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a
Y?u??R?i?`??????Unknown
e$Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a
Y?u??R?i?>?	?????Unknown?
V%HostMean"Mean(1      @9      @A      @I      @a
Y?u??R?iV*h?????Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a
Y?u??R?i?d?|????Unknown
?'HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a
Y?u??R?i?ן$?????Unknown
~(HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a
Y?u??R?i]?ڂw????Unknown
b)HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a
Y?u??R?i
???????Unknown
?*HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a
Y?u??R?i?pP?r????Unknown
?+HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a
Y?u??R?idN???????Unknown
y,HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a
Y?u??R?i,??l????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a???xL?iRRr?????Unknown
\.HostGreater"Greater(1      @9      @A      @I      @a???xL?i?x	?????Unknown
s/HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a???xL?iԞ??????Unknown
z0HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???xL?i?v????Unknown
v1HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a???xL?iV?"???Unknown
v2HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???xL?i??#!???Unknown
~3HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a???xL?i?7{*????Unknown
?4HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???xL?i^'1]#???Unknown
?5HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???xL?iZ??7{*???Unknown
}6HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a???xL?i??>?1???Unknown
?7HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???xL?i??+E?8???Unknown
?8HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a???xL?i??K?????Unknown
?9HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???xL?i^?R?F???Unknown
?:HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a???xL?i?C0YN???Unknown
?;HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???xL?i?i?_/U???Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a
Y?u??B?i????Y???Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a
Y?u??B?i?G??^???Unknown
V>HostCast"Cast(1       @9       @A       @I       @a
Y?u??B?ib?4mkc???Unknown
X?HostCast"Cast_3(1       @9       @A       @I       @a
Y?u??B?i8%R*h???Unknown
X@HostCast"Cast_5(1       @9       @A       @I       @a
Y?u??B?i?o??l???Unknown
XAHostEqual"Equal(1       @9       @A       @I       @a
Y?u??B?i??z?q???Unknown
?BHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      T@9      T@A       @I       @a
Y?u??B?i?q?)fv???Unknown
uCHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a
Y?u??B?i????${???Unknown
?DHostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1       @9       @A       @I       @a
Y?u??B?ifO??????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a
Y?u??B?i<?7?????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a
Y?u??B?i- ?`????Unknown
rGHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a
Y?u??B?i??=?????Unknown
vHHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a
Y?u??B?i?
[Dޒ???Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a
Y?u??B?i?yx??????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a
Y?u??B?ij蕢[????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a
Y?u??B?i@W?Q????Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a
Y?u??B?i?? ٥???Unknown
wMHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a
Y?u??B?i?4????Unknown
yNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a
Y?u??B?i£_V????Unknown
xOHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a
Y?u??B?i?)????Unknown
?PHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a
Y?u??B?in?F?Ӹ???Unknown
?QHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a
Y?u??B?iD?cl?????Unknown
?RHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a
Y?u??B?i_?Q????Unknown
?SHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a
Y?u??B?i?͞?????Unknown
?THostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a
Y?u??B?i?<?y?????Unknown
?UHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a
Y?u??B?i???(?????Unknown
?VHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a
Y?u??B?ir??K????Unknown
?WHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a
Y?u??B?iH??
????Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a
Y?u??2?i?@??i????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a
Y?u??2?i?16?????Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??a
Y?u??2?i????(????Unknown
|[HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a
Y?u??2?i?fO??????Unknown
?\HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a
Y?u??2?i_?<?????Unknown
?]HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a
Y?u??2?i??l?F????Unknown
?^Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a
Y?u??2?i5????????Unknown
?_HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a
Y?u??2?i?D?C????Unknown
?`HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a
Y?u??2?i??d????Unknown
?aHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a
Y?u??2?iv????????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a
Y?u??2?i?j6J#????Unknown
?cHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a
Y?u??2?iL"š?????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a
Y?u??2?i??S??????Unknown
~eHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a
Y?u??2?i"??PA????Unknown
}fHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a
Y?u??2?i?Hq??????Unknown
gHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a
Y?u??2?i?????????Unknown
IhHostAssignAddVariableOp"AssignAddVariableOp_1(i?????????Unknown
JiHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
[jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown*?f
sHostDataset"Iterator::Model::ParallelMapV2(1     @R@9     @R@A     @R@I     @R@a${?ґ??i${?ґ???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?S@9     ?S@A     ?Q@I     ?Q@a??]-n¼?iT:?g *???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      A@9      A@A      A@I      A@a7a~W???iD?#{???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      >@9      >@A      :@I      :@a *?3??iQ?Ȟ?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      4@9      4@A      4@I      4@ah *?3??i^-n??????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      2@9      2@A      2@I      2@aT:?g *??iQ?Ȟ????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      0@9      0@A      0@I      0@a?3?????iAT:?g ???Unknown
^HostGatherV2"GatherV2(1      .@9      .@A      .@I      .@a?0???M??iKG?D????Unknown
`	HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a?0???M??iU:?g *???Unknown
V
HostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a^-n?????i+?3????Unknown
dHostDataset"Iterator::Model(1     @U@9     @U@A      (@I      (@a?&??jq??i????&????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      (@9      (@A      (@I      (@a?&??jq??iQ?Ȟ????Unknown
oHostSigmoid"sequential/dense_2/Sigmoid(1      (@9      (@A      (@I      (@a?&??jq??i;?g *???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a?#{?ґ?iX??0?????Unknown?
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@ah *?3??i[܄?]-???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      $@9      $@A      $@I      $@ah *?3??i^-n??????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      "@9      "@A      "@I      "@aT:?g *??iG?D?#???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a?3?????i7a~W????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?3?????i?w??	????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a^-n?????i?0???M???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a^-n?????iO?Ȟ?????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a^-n?????i?ґ=???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      ]@9      ]@A      @I      @a?&??jq??i?ґ=Q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?&??jq??i<Q?Ȟ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?&??jq??i?3??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?&??jq??itd?@T:???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?&??jq??i???????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a?&??jq??i??M??????Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @ah *?3??i.n??????Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @ah *?3??i?7a~W???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @ah *?3??i2???M????Unknown
? HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ah *?3??i?g *????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?3???y?i?JG????Unknown
?"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?3???y?i??td?@???Unknown
e#Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?3???y?i?Ȟ??t???Unknown?
V$HostMean"Mean(1      @9      @A      @I      @a?3???y?iT?Ȟ?????Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?3???y?i?	??Z????Unknown
?&HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?3???y?i$*?3???Unknown
~'HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?3???y?i?JG?D???Unknown
b(HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?3???y?i?jq?w???Unknown
?)HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?3???y?i\??0?????Unknown
?*HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?3???y?iī?M?????Unknown
y+HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a?3???y?i,??jq???Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?&??jqs?izd?@T:???Unknown
\-HostGreater"Greater(1      @9      @A      @I      @a?&??jqs?i???7a???Unknown
s.HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?&??jqs?i???????Unknown
z/HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?&??jqs?id-n??????Unknown
v0HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?&??jqs?i??M??????Unknown
v1HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?&??jqs?i ^-n?????Unknown
~2HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?&??jqs?iN?D?#???Unknown
?3HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?&??jqs?i????J???Unknown
?4HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?&??jqs?i?&??jq???Unknown
}5HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?&??jqs?i8???M????Unknown
?6HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?&??jqs?i?W??0????Unknown
?7HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a?&??jqs?i??jq????Unknown
?8HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?&??jqs?i"?JG????Unknown
?9HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?&??jqs?ip *?3???Unknown
?:HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?&??jqs?i??	??Z???Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?3???i?i?Ȟ??t???Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?3???i?i&?3?????Unknown
V=HostCast"Cast(1       @9       @A       @I       @a?3???i?iZ?Ȟ?????Unknown
X>HostCast"Cast_3(1       @9       @A       @I       @a?3???i?i??]-n????Unknown
X?HostCast"Cast_5(1       @9       @A       @I       @a?3???i?i?	??Z????Unknown
X@HostEqual"Equal(1       @9       @A       @I       @a?3???i?i??JG????Unknown
?AHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      T@9      T@A       @I       @a?3???i?i**?3???Unknown
uBHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a?3???i?i^:?g *???Unknown
?CHostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1       @9       @A       @I       @a?3???i?i?JG?D???Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?3???i?i?Z܄?]???Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?3???i?i?jq?w???Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?3???i?i.{?ґ???Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?3???i?ib??0?????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?3???i?i??0??????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?3???i?iʫ?M?????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?3???i?i??Z܄????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?3???i?i2??jq???Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?3???i?if܄?]-???Unknown
yMHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a?3???i?i???JG???Unknown
xNHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?3???i?i???7a???Unknown
?OHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?3???i?iD?#{???Unknown
?PHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a?3???i?i6?3????Unknown
?QHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?3???i?ij-n??????Unknown
?RHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?3???i?i?=Q?????Unknown
?SHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?3???i?i?M???????Unknown
?THost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?3???i?i^-n?????Unknown
?UHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?3???i?i:n??????Unknown
?VHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?3???i?in~W??0???Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??a?3???Y?i??ґ=???Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??a?3???Y?i????J???Unknown?
TYHostMul"Mul(1      ??9      ??A      ??I      ??a?3???Y?i?7a~W???Unknown
|ZHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?3???Y?i֞??td???Unknown
?[HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?3???Y?i?&??jq???Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?3???Y?i
?7a~???Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?3???Y?i$7a~W????Unknown
?^HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?3???Y?i>???M????Unknown
?_HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?3???Y?iXG?D????Unknown
?`HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?3???Y?ir?@T:????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?3???Y?i?W??0????Unknown
?bHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?3???Y?i????&????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?3???Y?i?g *????Unknown
~dHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a?3???Y?i??jq????Unknown
}eHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a?3???Y?i?w??	????Unknown
fHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a?3???Y?i     ???Unknown
IgHostAssignAddVariableOp"AssignAddVariableOp_1(i     ???Unknown
JhHostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown2CPU