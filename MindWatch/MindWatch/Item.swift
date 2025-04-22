//
//  Item.swift
//  MindWatch
//
//  Created by Nico on 4/22/25.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
