"?e
BHostIDLE"IDLE1    ???@A    ???@a?sJ?2???i?sJ?2????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     (?@9     (?@A     (?@I     (?@a?$ppf???i`xX?c???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?B@9     ?B@A      @@I      @@a?Yr%\i?i??}??|???Unknown
tHost_FusedMatMul"sequential_6/dense_19/Relu(1      :@9      :@A      :@I      :@a??l??cd?i?W\??????Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a?UA??c?i??YĖ????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      4@9      4@A      4@I      4@a???.s^_?iq ??E????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      2@9      2@A      2@I      2@a?? ?g;\?i?Ʊc????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      2@9      2@A      2@I      2@a?? ?g;\?i!?e?????Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      1@9      1@A      1@I      1@a????Z?i??V?????Unknown
~
HostMatMul"*gradient_tape/sequential_6/dense_20/MatMul(1      ,@9      ,@A      ,@I      ,@aPĠP?U?i?g???????Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a???Y?@Q?i+G?^q????Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      &@9      &@A      &@I      &@a???Y?@Q?iz&9?????Unknown
?HostMatMul",gradient_tape/sequential_6/dense_20/MatMul_1(1      $@9      $@A      $@I      $@a???.s^O?i6?[????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?? ?g;L?i_b?4????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?? ?g;L?i??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?? ?g;L?i?r?????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?? ?g;L?i????$???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?Yr%\I?ipW??j$???Unknown
dHostDataset"Iterator::Model(1      :@9      :@A       @I       @a?Yr%\I?i????*???Unknown
~HostMatMul"*gradient_tape/sequential_6/dense_19/MatMul(1       @9       @A       @I       @a?Yr%\I?i???0???Unknown
~HostMatMul"*gradient_tape/sequential_6/dense_21/MatMul(1       @9       @A       @I       @a?Yr%\I?i2m?=7???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aPĠP?E?i6??r?<???Unknown?
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aPĠP?E?i:?$?7B???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aPĠP?E?i> M?G???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aPĠP?E?iB1uo2M???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aPĠP?E?iFb?ïR???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aPĠP?E?iJ??-X???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aPĠP?E?iN??k?]???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a ?E?B?i??4?^b???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a ?E?B?i0?{?g???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a ?E?B?i????k???Unknown
~ HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a ?E?B?i?	?|p???Unknown
?!HostBiasAddGrad"7gradient_tape/sequential_6/dense_21/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ?E?B?i??PB1u???Unknown
?"HostReadVariableOp"+sequential_6/dense_19/MatMul/ReadVariableOp(1      @9      @A      @I      @a ?E?B?i?????y???Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a???.s^??iҾ???}???Unknown
v$HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a???.s^??i??cp?????Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???.s^??i?r?>?????Unknown
?&HostBiasAddGrad"7gradient_tape/sequential_6/dense_20/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???.s^??ilL/?????Unknown
t'Host_FusedMatMul"sequential_6/dense_20/Relu(1      @9      @A      @I      @a???.s^??iJ&?ۀ????Unknown
w(Host_FusedMatMul"sequential_6/dense_21/BiasAdd(1      @9      @A      @I      @a???.s^??i( ??l????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?Yr%\9?is???????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?Yr%\9?i?\??????Unknown
V+HostMean"Mean(1      @9      @A      @I      @a?Yr%\9?i	??՚???Unknown
s,HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?Yr%\9?iT???????Unknown
v-HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?Yr%\9?i?g??????Unknown
v.HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?Yr%\9?i??>????Unknown
~/HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?Yr%\9?i5ě?a????Unknown
?0HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?Yr%\9?i?r ?????Unknown
?1HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?Yr%\9?i? ??????Unknown
?2HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?Yr%\9?i?)˰???Unknown
?3HostBiasAddGrad"7gradient_tape/sequential_6/dense_19/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Yr%\9?ia}?(?????Unknown
?4HostMatMul",gradient_tape/sequential_6/dense_21/MatMul_1(1      @9      @A      @I      @a?Yr%\9?i?+34????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a ?E?2?id??|k????Unknown
X6HostCast"Cast_3(1      @9      @A      @I      @a ?E?2?i1z?Ż???Unknown
\7HostGreater"Greater(1      @9      @A      @I      @a ?E?2?iԳ ????Unknown
u8HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a ?E?2?i?6?Vz????Unknown
?9HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a ?E?2?iD?d??????Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a ?E?2?i?;?.????Unknown
`;HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a ?E?2?i???0?????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a ?E?2?ilAOy?????Unknown
?=Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a ?E?2?i$???=????Unknown
~>HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a ?E?2?i?F?
?????Unknown
??HostReluGrad",gradient_tape/sequential_6/dense_19/ReluGrad(1      @9      @A      @I      @a ?E?2?i??9S?????Unknown
?@HostReadVariableOp",sequential_6/dense_19/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a ?E?2?iLLݛL????Unknown
rAHostSigmoid"sequential_6/dense_21/Sigmoid(1      @9      @A      @I      @a ?E?2?iπ??????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?Yr%\)?i*&Cj8????Unknown
VCHostCast"Cast(1       @9       @A       @I       @a?Yr%\)?iP}??????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @a?Yr%\)?iv??u[????Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?Yr%\)?i?+???????Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?Yr%\)?iL?~????Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?Yr%\)?i??????Unknown
zHHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?Yr%\)?i1ь?????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?Yr%\)?i4??3????Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?Yr%\)?iZ?U??????Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?Yr%\)?i?6V????Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?Yr%\)?i??ڣ?????Unknown
?MHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?Yr%\)?i???)y????Unknown
?NHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?Yr%\)?i?;_?
????Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?Yr%\)?i?!5?????Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?Yr%\)?i>???-????Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?Yr%\)?idA?@?????Unknown
?RHostReadVariableOp",sequential_6/dense_20/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?Yr%\)?i??h?P????Unknown
?SHostReadVariableOp"+sequential_6/dense_20/MatMul/ReadVariableOp(1       @9       @A       @I       @a?Yr%\)?i??*L?????Unknown
?THostReadVariableOp",sequential_6/dense_21/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?Yr%\)?i?F??s????Unknown
?UHostReadVariableOp"+sequential_6/dense_21/MatMul/ReadVariableOp(1       @9       @A       @I       @a?Yr%\)?i???W????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?Yr%\?i?ɐ?????Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??a?Yr%\?i"?qݖ????Unknown
XXHostCast"Cast_5(1      ??9      ??A      ??I      ??a?Yr%\?i? S?_????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a?Yr%\?iHL4c(????Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??a?Yr%\?i?w&?????Unknown
d[HostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?Yr%\?in????????Unknown
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?Yr%\?i?׫?????Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?Yr%\?i???nK????Unknown
w^HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?Yr%\?i'&?1????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?Yr%\?i?Q{??????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?Yr%\?iM}\??????Unknown
?aHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?Yr%\?i??=zn????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?Yr%\?is?=7????Unknown
?cHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?Yr%\?i     ???Unknown
?dHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?Yr%\?i̕pad ???Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?Yr%\?i?+??? ???Unknown
?fHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?Yr%\?i^?Q$-???Unknown
?gHostReluGrad",gradient_tape/sequential_6/dense_20/ReluGrad(1      ??9      ??A      ??I      ??a?Yr%\?i'W????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i'W????Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     (?@9     (?@A     (?@I     (?@a5?.;K??i5?.;K???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?B@9     ?B@A      @@I      @@a???????iA;????Unknown
tHost_FusedMatMul"sequential_6/dense_19/Relu(1      :@9      :@A      :@I      :@a???m???iK???????Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a3A?L-??iUqW?I???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      4@9      4@A      4@I      4@a??z????i]YB??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      2@9      2@A      2@I      2@a?A;???id]/?=2???Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      2@9      2@A      2@I      2@a?A;???ika??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      1@9      1@A      1@I      1@a????S??irs?B	???Unknown
~	HostMatMul"*gradient_tape/sequential_6/dense_20/MatMul(1      ,@9      ,@A      ,@I      ,@aXO?&z??iw?{??^???Unknown
l
HostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a???߀?i{??k????Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      &@9      &@A      &@I      &@a???߀?i{d??????Unknown
?HostMatMul",gradient_tape/sequential_6/dense_20/MatMul_1(1      $@9      $@A      $@I      $@a??z??~?i??Y?H#???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?A;?{?i?q?ǂZ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?A;?{?i??F⼑???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?A;?{?i?u???????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?A;?{?i??31 ???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?????x?i??+KH1???Unknown
dHostDataset"Iterator::Model(1      :@9      :@A       @I       @a?????x?i?#_b???Unknown
~HostMatMul"*gradient_tape/sequential_6/dense_19/MatMul(1       @9       @A       @I       @a?????x?i???v????Unknown
~HostMatMul"*gradient_tape/sequential_6/dense_21/MatMul(1       @9       @A       @I       @a?????x?i?7??????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aXO?&zu?i?Պ4?????Unknown?
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aXO?&zu?i?s?v???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aXO?&zu?i?|?jE???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aXO?&zu?i???_p???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aXO?&zu?i?MmjS????Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aXO?&zu?i????G????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aXO?&zu?i??^<????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a'?|?hr?i?5Xl???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a'?|?hr?i??Q??:???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a'?|?hr?i??K:?_???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a'?|?hr?i?9E??????Unknown
? HostBiasAddGrad"7gradient_tape/sequential_6/dense_21/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a'?|?hr?i??>S????Unknown
?!HostReadVariableOp"+sequential_6/dense_19/MatMul/ReadVariableOp(1      @9      @A      @I      @a'?|?hr?i??8o$????Unknown
?"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a??z??n?i?K???????Unknown
v#HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??z??n?i?.p????Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??z??n?i¿??/*???Unknown
?%HostBiasAddGrad"7gradient_tape/sequential_6/dense_20/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??z??n?i?y#q?H???Unknown
t&Host_FusedMatMul"sequential_6/dense_20/Relu(1      @9      @A      @I      @a??z??n?i?3???g???Unknown
w'Host_FusedMatMul"sequential_6/dense_21/BiasAdd(1      @9      @A      @I      @a??z??n?i??r;????Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?????h?iʵǞ???Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?????h?i?}?R????Unknown
V*HostMean"Mean(1      @9      @A      @I      @a?????h?i?E@?????Unknown
s+HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?????h?i??i????Unknown
v,HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?????h?i??t? ???Unknown
v-HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?????h?iԝ?????Unknown
~.HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?????h?i?e??2???Unknown
?/HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?????h?i?-?A?J???Unknown
?0HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?????h?i????#c???Unknown
?1HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?????h?iܽ?u?{???Unknown
?2HostBiasAddGrad"7gradient_tape/sequential_6/dense_19/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?????h?iޅ?;????Unknown
?3HostMatMul",gradient_tape/sequential_6/dense_21/MatMul_1(1      @9      @A      @I      @a?????h?i?M??Ƭ???Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a'?|?hb?i?#c]/????Unknown
X5HostCast"Cast_3(1      @9      @A      @I      @a'?|?hb?i????????Unknown
\6HostGreater"Greater(1      @9      @A      @I      @a'?|?hb?i??\? ????Unknown
u7HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a'?|?hb?i???wi????Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a'?|?hb?i?{V+????Unknown
v9HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a'?|?hb?i?Q??:???Unknown
`:HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a'?|?hb?i?'P??-???Unknown
b;HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a'?|?hb?i???E@???Unknown
?<Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a'?|?hb?i??I?tR???Unknown
~=HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a'?|?hb?i??Ƭ?d???Unknown
?>HostReluGrad",gradient_tape/sequential_6/dense_19/ReluGrad(1      @9      @A      @I      @a'?|?hb?i?C`Fw???Unknown
??HostReadVariableOp",sequential_6/dense_19/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a'?|?hb?i?U??????Unknown
r@HostSigmoid"sequential_6/dense_21/Sigmoid(1      @9      @A      @I      @a'?|?hb?i?+=?????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?????X?i?;?]????Unknown
VBHostCast"Cast(1       @9       @A       @I       @a?????X?i??8a?????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @a?????X?i??6.?????Unknown
jDHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?????X?i??4?.????Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?????X?i??2?t????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?????X?i??0??????Unknown
zGHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?????X?i?g.b ????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?????X?i?K,/F????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?????X?i?/*??
???Unknown
?JHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?????X?i?(?????Unknown
xKHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?????X?i??%?#???Unknown
?LHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?????X?i??#c]/???Unknown
?MHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?????X?i??!0?;???Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?????X?i????G???Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?????X?i???.T???Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?????X?i?k?t`???Unknown
?QHostReadVariableOp",sequential_6/dense_20/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?????X?i?Od?l???Unknown
?RHostReadVariableOp"+sequential_6/dense_20/MatMul/ReadVariableOp(1       @9       @A       @I       @a?????X?i?31 y???Unknown
?SHostReadVariableOp",sequential_6/dense_21/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?????X?i ?E????Unknown
?THostReadVariableOp"+sequential_6/dense_21/MatMul/ReadVariableOp(1       @9       @A       @I       @a?????X?i?ˋ????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?????H?i?????Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??a?????H?i??ѝ???Unknown
XWHostCast"Cast_5(1      ??9      ??A      ??I      ??a?????H?iҏ~?????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??a?????H?i?e????Unknown?
TYHostMul"Mul(1      ??9      ??A      ??I      ??a?????H?i??K:????Unknown
dZHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?????H?i?2]????Unknown
}[HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?????H?i???????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?????H?i?
??????Unknown
w]HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?????H?i~???????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?????H?ip??????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?????H?ib??????Unknown
?`HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?????H?iT?.????Unknown
?aHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?????H?iF?Q????Unknown
?bHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?????H?i8ft????Unknown
?cHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?????H?i*?L?????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?????H?i3?????Unknown
?eHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?????H?i??????Unknown
?fHostReluGrad",gradient_tape/sequential_6/dense_20/ReluGrad(1      ??9      ??A      ??I      ??a?????H?i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown2CPU